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
from src.filters import bandpass

seed_torch(seed=CFG.seed)


def inference(model, checkpoints, test_loader):
    device = get_device()
    model.to(device)
    states = [torch.load(cps) for cps in checkpoints]
    tk0 = tqdm(enumerate(test_loader), total=len(test_loader))
    probs = []
    for i, d in tk0:
        x = bandpass(d[0], **CFG.bandpass_params)
        images = torch.from_numpy(x).to(device)
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


def save_submission(input_dir, output_dir, predictions):
    df = pd.read_csv(f"{input_dir}/sample_submission.csv")
    df['target'] = predictions
    df[['id', 'target']].to_csv(f"{output_dir}/submission.csv", index=False)


def main():
    args = parse_args()
    checkpoints = args.checkpoints

    input_dir = '/content/input'
    output_dir = './output'

    model = CustomModel(CFG, pretrained=False)

    test_loader = TFRecordDataLoader(
        files=glob.glob(f"{input_dir}/test/test*.tfrecords"),
        batch_size=CFG.batch_size * 2,
        cache=True,
        train=False,
        repeat=False,
        shuffle=False,
        labeled=False,
        return_image_ids=False,
    )

    # inference
    predictions = inference(model, checkpoints, test_loader)

    # save submission.csv
    save_submission(input_dir, output_dir, predictions)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoints', type=str, required=True, dest='checkpoints')
    args = parser.parse_args()

    # change str to list
    checkpoints = args.checkpoints.split(',')
    for cps in checkpoints:
        if not os.path.exists(cps): raise Exception(f"{cps} is not found.")
    args.checkpoints = checkpoints

    return args


if __name__=='__main__':
    main()

