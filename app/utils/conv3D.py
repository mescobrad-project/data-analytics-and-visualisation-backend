# MRIs

import torch
import torch.nn as nn

class Conv3D(nn.Module):

    def __init__(self):
        super(Conv3D, self).__init__()

        self.num_labels = 2
        self.classifier = nn.Linear(512, self.num_labels)
        self.dropout = nn.Dropout(0.3)
        self.in_channels = 1  # because only FLAIR
        self.group1 = nn.Sequential(
            nn.Conv3d(self.in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 2, 2)))
        self.group2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        self.group3 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        self.group4 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        self.group5 = nn.Sequential(
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1)))

    def forward(self, x, labels=None):

        out = self.group1(x.float())
        out = self.group2(out)
        out = self.group3(out)
        out = self.group4(out)
        x = self.group5(out)
        y = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)
        logits = self.classifier(y)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return loss, logits if labels is not None else logits
