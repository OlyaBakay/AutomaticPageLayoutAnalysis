from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo
import torchvision.models


class SqueezeNet(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super(SqueezeNet, self).__init__()
        self.num_classes = num_classes

        model = torchvision.models.squeezenet1_0(pretrained=pretrained)
        classifier = list(model.classifier)
        final_conv = classifier[1]
        final_conv = nn.Conv2d(final_conv.in_channels, self.num_classes, kernel_size=1)
        classifier[1] = final_conv
        model.classifier = nn.Sequential(*classifier)
        self.backbone = model

    def forward(self, x):
        B, C, W, H = x.shape
        if C == 1:
            x = x.expand(B, 3, W, H)
        return self.backbone(x)


class UNet(nn.Module):
    CHECKPOINT_URL = 'https://github.com/mateuszbuda/brain-segmentation-pytorch/releases/download/v1.0/unet-e012d006.pt'

    def __init__(self, in_channels=3, out_channels=1, init_features=32, pretrained=True, **kwargs):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

        if pretrained:
            self.load_state_dict(torch.utils.model_zoo.load_url(self.CHECKPOINT_URL))

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return self.conv(dec1)

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )


class Fast1D(nn.Module):
    def __init__(self, outputs):
        super(Fast1D, self).__init__()
        self.conv1_x = nn.Conv1d(1, 50, 3)
        self.conv2_x = nn.Conv1d(50, 50, 3)
        self.conv3_x = nn.Conv1d(50, 50, 3)
        self.dp1_x = nn.Dropout(0.1)
        self.dp2_x = nn.Dropout(0.1)
        self.dp3_x = nn.Dropout(0.1)

        self.conv1_y = nn.Conv1d(1, 50, 3)
        self.conv2_y = nn.Conv1d(50, 50, 3)
        self.conv3_y = nn.Conv1d(50, 50, 3)
        self.dp1_y = nn.Dropout(0.1)
        self.dp2_y = nn.Dropout(0.1)
        self.dp3_y = nn.Dropout(0.1)

        self.dp4_conc = nn.Dropout(0.3)

        self.fc1 = nn.Linear(1000, 50)
        self.fc2 = nn.Linear(50, outputs)

    def forward(self, X):
        # print(X.shape)
        x, y = X[:, 0], X[:, 1]
        # Max pooling over a (2, 2) window
        x = F.max_pool1d(F.relu(self.conv1_x(x)), 2)
        x = self.dp1_x(x)
        x = F.max_pool1d(F.relu(self.conv2_x(x)), 2)
        x = self.dp2_x(x)
        x = F.max_pool1d(F.relu(self.conv3_x(x)), 2)
        x = self.dp3_x(x)

        y = F.max_pool1d(F.relu(self.conv1_y(y)), 2)
        y = self.dp1_y(y)
        y = F.max_pool1d(F.relu(self.conv2_y(y)), 2)
        y = self.dp2_y(y)
        y = F.max_pool1d(F.relu(self.conv3_y(y)), 2)
        y = self.dp3_y(y)

        conc = torch.cat((x, y), 1)
        conc = conc.reshape(conc.size(0), -1)
        conc = F.relu(self.fc1(conc))
        conc = self.dp4_conc(conc)
        conc = self.fc2(conc)
        if self.training == False:
            _, conc = torch.max(conc, 1)
        return conc
