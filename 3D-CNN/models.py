import torch
import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 48, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(48, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(128, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(192, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(192*4*4*4,192),  # 256*dimision*2/16=32*dimision
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(192, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 2),
        )
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x



class Mnist(nn.Module):

    def __init__(self):
        super(Mnist, self).__init__()

        self.features = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),

            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # nn.MaxPool3d(kernel_size=2, stride=2),

            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128*1*1*1, 128),  # 256*dimision*2/16=32*dimision
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        # x = self.avg_pool(x)
        # x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x



class FCN(nn.Module):

    def __init__(self):
        super(FCN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )
        self.conv3d = nn.Conv3d(32, 2, kernel_size=1, stride=1, padding=0)
        self.avg_pool3d = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        x = self.features(x)
        x = self.conv3d(x)
        x = self.avg_pool3d(x)
        x = torch.squeeze(x)
        return x



