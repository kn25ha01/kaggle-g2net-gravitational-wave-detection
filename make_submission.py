import os
import glob
import time
import argparse
from pathlib import Path
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
import scipy as sp
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CFG
from src.utils import seed_torch, get_device
from src.model_factory import CustomModel
from src.dataset_factory import TFRecordDataLoader


SAVEDIR = Path(CFG.output_dir)
seed_torch(seed=CFG.seed)
device = get_device()


def inference(model, states, test_loader, device):
    model.to(device)
    tk0 = tqdm(enumerate(test_loader), total=len(test_loader))
    probs = []
    for i, d in tk0:
        images = torch.from_numpy(d[0]).to(device)
        avg_preds = []
        for state in states:
            model.load_state_dict(state['model'])
            model.eval()
            with torch.no_grad():
                y_preds = model(images)
            avg_preds.append(y_preds.sigmoid().to('cpu').numpy())
        avg_preds = np.mean(avg_preds, axis=0)
        probs.append(avg_preds)
    probs = np.concatenate(probs)
    return probs


def main(args):
    input_dir = args.input_dir
    checkpoints = args.checkpoints

    if CFG.test:
        model = CustomModel(CFG, pretrained=False)

        states = [torch.load(cps) for cps in checkpoints]

        test_files = glob.glob(f"{input_dir}/test/test*.tfrecords")

        test_loader = TFRecordDataLoader(
            test_files,
            batch_size=CFG.batch_size * 2,
            cache=True,
            train=False,
            repeat=False,
            shuffle=False,
            labeled=False,
            return_image_ids=False,
        )

        predictions = inference(model, states, test_loader, device)

        # save result
        test_df = pd.read_csv(f"{input_dir}/sample_submission.csv")
        test_df['target'] = predictions
        test_df[['id', 'target']].to_csv(f"{SAVEDIR}/submission.csv", index=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, dest='input_dir')
    parser.add_argument('-c', '--checkpoints', type=str, required=True, dest='checkpoints')
    args = parser.parse_args()
    if not os.path.exists(args.input_dir): raise Exception(f"{args.input_dir} is not found.")

    # to list
    checkpoints = args.checkpoints.split(',')
    for cps in checkpoints:
        if not os.path.exists(cps): raise Exception(f"{cps} is not found.")
    args.checkpoints = checkpoints

    return args


if __name__=='__main__':
    args = parse_args()
    main(args)

