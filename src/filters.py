import numpy as np
from scipy import signal


def moving_average(x, window_size=7):
    pad_size = window_size // 2
    x = np.pad(x, [pad_size, pad_size], 'edge') # copy edge
    x = np.convolve(x, np.ones(window_size), 'valid') / window_size # none padding
    return x


def apply_moving_average(x, params):
    x[2] = moving_average(x[2], **params)
    return x


def bandpass(x, lf=20, hf=500, order=8, sr=2048):
    '''
    Cell 33 of https://www.gw-openscience.org/LVT151012data/LOSC_Event_tutorial_LVT151012.html
    https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
    https://www.kaggle.com/firefliesqn/g2net-gpu-newbie
    '''
    sos = signal.butter(order, [lf, hf], btype="bandpass", output="sos", fs=sr)
    normalization = np.sqrt((hf - lf) / (sr / 2))
    if x.ndim ==2:
        for i in range(3):
            x[i] = signal.sosfilt(sos, x[i]) * normalization
    elif x.ndim == 3: # batch
        for i in range(x.shape[0]):
            for j in range(3):
                x[i, j] = signal.sosfilt(sos, x[i, j]) * normalization
    return x


def apply_bandpass(x, params):
    x = bandpass(x, **params)
    return x


def apply_transforms(x, cfg):
    for k, v in cfg.preprocessing.items():
        if v["apply"]:
            if k == "moving_average":
                x = apply_moving_average(x, v["params"])
            if k == "bandpass":
                x = apply_bandpass(x, v["params"])
    return x

