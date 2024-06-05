import torchvision
import torch.nn as nn
from collections import OrderedDict

class ResNet18_3D(nn.Module):

    def __init__(self):
        super(ResNet18_3D, self).__init__()

        self.resnet = torchvision.models.video.r3d_18(pretrained=True)

        # Modify the first convolution layer to accept 1 channel instead of 3
        self.resnet.stem[0] = nn.Conv3d(1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)

        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 2)

        '''
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
        '''

        layers_to_freeze = [self.resnet.layer1[0], self.resnet.layer1[1],
                            self.resnet.layer2[0], self.resnet.layer2[1],
                            self.resnet.layer3[0], self.resnet.layer3[1],
                            self.resnet.layer4[0]]#, self.resnet.layer4[1].conv1]

        for child in layers_to_freeze:
            for param in child.parameters():
                param.requires_grad = False

        print('No. of trainable params', sum(p.numel() for p in self.resnet.parameters() if p.requires_grad))

    def forward(self, x, labels=None):
        outputs = self.resnet(x)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(outputs, labels)

        return (loss, outputs) if labels is not None else (None, outputs)
