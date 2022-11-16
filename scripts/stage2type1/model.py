import timm
import torch
from config import cfg
import torch.nn as nn

class TimmModel(nn.Module):
    def __init__(self, backbone, pretrained=False):
        super(TimmModel, self).__init__()

        self.encoder = timm.create_model(
            backbone,
            in_chans=cfg.in_chans,
            num_classes=cfg.out_dim,
            features_only=False,
            drop_rate=cfg.drop_rate,
            drop_path_rate=cfg.drop_path_rate,
            pretrained=pretrained
        )

        if 'efficient' in backbone:
            hdim = self.encoder.conv_head.out_channels
            self.encoder.classifier = nn.Identity()
        elif 'convnext' in backbone:
            hdim = self.encoder.head.fc.in_features
            self.encoder.head.fc = nn.Identity()


        self.lstm = nn.LSTM(hdim, 256, num_layers=2, dropout=cfg.drop_rate, bidirectional=True, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(cfg.drop_rate_last),
            nn.LeakyReLU(0.1),
            nn.Linear(256, cfg.out_dim),
        )

    def forward(self, x):  # (bs, nslice, ch, sz, sz)
        bs = x.shape[0]
        x = x.view(bs * cfg.n_slice_per_c, cfg.in_chans, cfg.image_size, cfg.image_size)
        feat = self.encoder(x)
        feat = feat.view(bs, cfg.n_slice_per_c, -1)
        feat, _ = self.lstm(feat)
        feat = feat.contiguous().view(bs * cfg.n_slice_per_c, -1)
        feat = self.head(feat)
        feat = feat.view(bs, cfg.n_slice_per_c).contiguous()

        return feat

m = TimmModel(cfg.backbone)
m(torch.rand(2, cfg.n_slice_per_c, cfg.in_chans, cfg.image_size, cfg.image_size)).shape