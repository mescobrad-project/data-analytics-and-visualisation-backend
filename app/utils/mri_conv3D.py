# MRIs

import torch
import torch.nn as nn

class Conv3D(nn.Module):

    def __init__(self):
        super(Conv3D, self).__init__()

        self.num_labels = 2
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
            nn.Dropout(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        self.group5 = nn.Sequential(
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.Dropout(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        self.group6 = nn.Sequential(
            nn.Conv3d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm3d(1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.AdaptiveAvgPool3d((1, 1, 1)))

        # self.dense = nn.Linear(1024, 128)
        self.classifier = nn.Linear(1024, self.num_labels)

    def forward(self, x, labels=None):

        out = self.group1(x.float())
        out = self.group2(out)
        out = self.group3(out)
        out = self.group4(out)
        out = self.group5(out)
        out = self.group6(out)
        #y = torch.mean(out.view(out.size(0), out.size(1), -1), dim=2)
        #y = self.dense(y)
        out = out.view(out.size(0), -1) #flatten the output of averages
        logits = self.classifier(out)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        #return loss, logits if labels is not None else logits
        return (loss, logits) if labels is not None else (None, logits)
