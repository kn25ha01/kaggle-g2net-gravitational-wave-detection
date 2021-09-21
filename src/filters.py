import numpy as np
from scipy import signal


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
