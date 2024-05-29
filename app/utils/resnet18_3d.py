import torchvision
import torch.nn as nn
from collections import OrderedDict

class ResNet18_3D(nn.Module):

    def __init__(self,
                 trainable_feature_layers=None):
        super(ResNet18_3D, self).__init__()

        resnet = torchvision.models.video.r3d_18(pretrained=True)
        resnet.fc = nn.Linear(resnet.fc.in_features, 2)

        layers = [
            ('stem', resnet.stem),
            ('layer1_0', resnet.layer1[0]),
            ('layer1_1', resnet.layer1[1]),
            ('layer2_0', resnet.layer2[0]),
            ('layer2_1', resnet.layer2[1]),
            ('layer3_0', resnet.layer3[0]),
            ('layer3_1', resnet.layer3[1]),
            ('layer4_0', resnet.layer4[0]),
            ('layer4_1', resnet.layer4[1]),
            ('adpt_avgpool', resnet.avgpool),
            ('clf', resnet.fc)
        ]

        self.model = nn.Sequential(OrderedDict(layers))

        # LAYERS TO FREEZE DURING TRAINING
        # 'adpt_avgpool' does not contain learnable params
        all_layers = [layer for name, layer in layers if name not in ['adpt_avgpool', 'clf']]

        if trainable_feature_layers is None:
            self.freeze = all_layers
        else:
            assert all(x in range(len(all_layers)) for x in
                       trainable_feature_layers), "Invalid layer indices in trainable_feature_layers"
            self.freeze = [layer for i, layer in enumerate(all_layers) if i not in trainable_feature_layers]

        for child in self.freeze:
            for param in child.parameters():
                param.requires_grad = False

    def trainable_params(self):
        print('No. of trainable params', sum(p.numel() for p in self.model.parameters() if p.requires_grad))

    def forward(self, x, labels=None):
        outputs = self.model(x)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(outputs, labels)

        return (loss, outputs) if labels is not None else (None, outputs)
