import torch
import torch.nn as nn
import timm
from nnAudio.Spectrogram import CQT1992v2


class CustomModel(nn.Module):
    def __init__(self, cfg, pretrained=False):
        super().__init__()
        self.cfg = cfg
        self.q_transform = CQT1992v2(**cfg.qtransform_params)
        self.model = timm.create_model(self.cfg.model_name, pretrained=pretrained, in_chans=cfg.in_chans)
        self.n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(self.n_features, self.cfg.target_size)

    def forward(self, waves):
        channels = [self.q_transform(waves[:, i]) for i in range(3)]
        image = torch.stack(channels, dim=1)
        output = self.model(image)
        return output

