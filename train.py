import os
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy as sp
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
import wandb

from config import CFG
from src.utils import get_score, init_logger, seed_torch, get_device
from src.model_factory import CustomModel
from src.dataset_factory import TFRecordDataLoader
from src.helper import AverageMeter, timeSince, max_memory_allocated


SAVEDIR = Path(CFG.output_dir)
if not os.path.exists(SAVEDIR):
    os.mkdir(SAVEDIR)

LOGGER = init_logger(log_file=CFG.log_filename)

seed_torch(seed=CFG.seed)
device = get_device()


def train_fn(files, model, criterion, optimizer, epoch, scheduler, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()

    # switch to train mode
    model.train()
    start = end = time.time()
    global_step = 0

    train_loader = TFRecordDataLoader(
        files, cache=True, batch_size=CFG.batch_size, shuffle=True)
    for step, d in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        images = torch.from_numpy(d[0]).to(device)
        labels = torch.from_numpy(d[1]).to(device)

        batch_size = labels.size(0)
        y_preds = model(images)
        loss = criterion(y_preds.view(-1), labels.view(-1))
        # record loss
        losses.update(loss.item(), batch_size)
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step % CFG.print_freq == 0:
            print('Epoch: [{0}/{1}][{2}/{3}] '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Grad: {grad_norm:.4f}  '
                  'LR: {lr:.6f}  '
                  'Elapsed: {remain:s} '
                  'Max mem: {mem:s}'
                  .format(
                   epoch+1, CFG.epochs, step, len(train_loader),
                   loss=losses,
                   grad_norm=grad_norm,
                   lr=scheduler.get_lr()[0],
                   remain=timeSince(start, float(step + 1) / len(train_loader)),
                   mem=max_memory_allocated()))
        # wandb
        wandb.log({
            f"loss": losses.val,
            f"lr": scheduler.get_lr()[0]
        })
    return losses.avg


def valid_fn(files, model, criterion, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()
    # switch to evaluation mode
    model.eval()
    filenames = []
    targets = []
    preds = []
    start = end = time.time()
    valid_loader = TFRecordDataLoader(
        files, cache=True, batch_size=CFG.batch_size * 2, shuffle=False)
    for step, d in enumerate(valid_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        targets.extend(d[1].reshape(-1).tolist())
        filenames.extend([f.decode("UTF-8") for f in d[2]])
        
        images = torch.from_numpy(d[0]).to(device)
        labels = torch.from_numpy(d[1]).to(device)

        batch_size = labels.size(0)
        # compute loss
        with torch.no_grad():
            y_preds = model(images)
        loss = criterion(y_preds.view(-1), labels.view(-1))
        losses.update(loss.item(), batch_size)

        preds.append(y_preds.sigmoid().to('cpu').numpy())
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step % CFG.print_freq == 0:
            print('EVAL: [{0}/{1}] '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  .format(
                   step, len(valid_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses,
                   remain=timeSince(start, float(step+1)/len(valid_loader)),
                   ))
    predictions = np.concatenate(preds).reshape(-1)
    return losses.avg, predictions, np.array(targets), np.array(filenames)


# ====================================================
# Train loop
# ====================================================
def train_loop(train_tfrecords: np.ndarray, val_tfrecords: np.ndarray, fold: int):
    
    LOGGER.info(f"========== fold: {fold} training ==========")
    
    # ====================================================
    # scheduler 
    # ====================================================
    def get_scheduler(optimizer):
        if CFG.scheduler=='ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=CFG.factor, patience=CFG.patience, verbose=True, eps=CFG.eps)
        elif CFG.scheduler=='CosineAnnealingLR':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.T_max, eta_min=CFG.min_lr, last_epoch=-1)
        elif CFG.scheduler=='CosineAnnealingWarmRestarts':
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=CFG.T_0, T_mult=1, eta_min=CFG.min_lr, last_epoch=-1)
        return scheduler

    # ====================================================
    # model & optimizer
    # ====================================================
    model = CustomModel(CFG, pretrained=True)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay, amsgrad=False)
    scheduler = get_scheduler(optimizer)

    # ====================================================
    # loop
    # ====================================================
    criterion = nn.BCEWithLogitsLoss()

    best_score = 0.
    best_loss = np.inf
    
    for epoch in range(CFG.epochs):
        
        start_time = time.time()
        
        # train
        avg_loss = train_fn(train_tfrecords, model, criterion, optimizer, epoch, scheduler, device)

        # eval
        avg_val_loss, preds, targets, files = valid_fn(val_tfrecords, model, criterion, device)
        valid_result_df = pd.DataFrame({"target": targets, "preds": preds, "id": files})
        
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(avg_val_loss)
        elif isinstance(scheduler, optim.lr_scheduler.CosineAnnealingLR):
            scheduler.step()
        elif isinstance(scheduler, optim.lr_scheduler.CosineAnnealingWarmRestarts):
            scheduler.step()

        # scoring
        score = get_score(targets, preds)

        elapsed = time.time() - start_time

        LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        LOGGER.info(f'Epoch {epoch+1} - Score: {score:.4f}')
        # wandb
        wandb.log({
            f"epoch": epoch + 1,
            f"avg_train_loss": avg_loss,
            f"avg_val_loss": avg_val_loss,
            f"score": score,
        })

        if score > best_score:
            best_score = score
            LOGGER.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
            torch.save({'model': model.state_dict(), 
                        'preds': preds},
                        SAVEDIR / f'{CFG.model_name}_fold{fold}_best_score.pth')
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            LOGGER.info(f'Epoch {epoch+1} - Save Best Loss: {best_loss:.4f} Model')
            torch.save({'model': model.state_dict(), 
                        'preds': preds},
                        SAVEDIR / f'{CFG.model_name}_fold{fold}_best_loss.pth')
    
    valid_result_df["preds"] = torch.load(SAVEDIR / f"{CFG.model_name}_fold{fold}_best_loss.pth",
                                          map_location="cpu")["preds"]

    return valid_result_df


def class2dict(f):
    return dict((name, getattr(f, name)) for name in dir(f) if not name.startswith('__'))


def main(args):
    # get input dir
    INPUT_DIR = args.input_dir
    exp_num = args.exp_num

    fold0 = [INPUT_DIR + f"/train_fold01/train{i}.tfrecords" for i in range(0, 5)]
    fold1 = [INPUT_DIR + f"/train_fold01/train{i}.tfrecords" for i in range(5, 10)]
    fold2 = [INPUT_DIR + f"/train_fold23/train{i}.tfrecords" for i in range(10, 15)]
    fold3 = [INPUT_DIR + f"/train_fold23/train{i}.tfrecords" for i in range(15, 20)]
    folds = [fold0, fold1, fold2, fold3]

    def get_result(result_df):
        preds = result_df['preds'].values
        labels = result_df[CFG.target_col].values
        score = get_score(labels, preds)
        LOGGER.info(f'Score: {score:<.4f}')

    # wandb
    wandb_api_key = args.wandb_api_key
    wandb.login(key=wandb_api_key)
    
    if CFG.train:
        oof_df = pd.DataFrame()

        for fold in range(CFG.n_fold):
            if fold in CFG.trn_fold:
                # wandb
                wandb.init(
                    project=CFG.project,
                    name=f"{exp_num}_fold{fold}",
                    config=class2dict(CFG),
                    group=CFG.model_name,
                    job_type="train",
                )

                train_files = [f for i, f in enumerate(folds) if i != fold]
                train_files = [i for j in train_files for i in j]
                valid_files = folds[fold]
                _oof_df = train_loop(train_files, valid_files, fold)
                oof_df = pd.concat([oof_df, _oof_df])
                LOGGER.info(f"========== fold: {fold} result ==========")
                get_result(_oof_df)
        # CV result
        LOGGER.info(f"========== CV ==========")
        get_result(oof_df)
        # save result
        oof_df.to_csv(SAVEDIR / 'oof_df.csv', index=False)

    # wandb
    wandb.finish()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, dest='input_dir')
    parser.add_argument('-e', '--exp_num', type=str, required=True, dest='exp_num')
    parser.add_argument('--wandb_api_key', type=str, required=True, dest='wandb_api_key')
    args = parser.parse_args()
    if not os.path.exists(args.input_dir): raise Exception(f"{args.input_dir} is not found.")
    return args


if __name__=='__main__':
    args = parse_args()
    main(args)

