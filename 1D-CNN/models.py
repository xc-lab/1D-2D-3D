import torch
import torch.nn as nn
import torch.nn.functional as F


class Mnist(nn.Module):
    def __init__(self, in_channels=1, n_classes=2):
        super(Mnist, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Linear(32*5*8, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(32, n_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x



# class AlexNet(nn.Module):
#     def __init__(self):
#         super(AlexNet, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(1, 48, kernel_size=(5,1), stride=(2,1), padding=(2,0)),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=(2,1), stride=(2,1)),
#             nn.Conv2d(48, 128, kernel_size=(5,1), stride=(2,1), padding=(2,0)),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=(2,1), stride=(2,1)),
#             nn.Conv2d(128, 192, kernel_size=(3,1), stride=(1,1), padding=(1,0)),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(192, 192, kernel_size=(3,1), stride=(1,1), padding=(1,0)),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=(2,1), stride=(2,1)),
#         )
#         self.classifier = nn.Sequential(
#             nn.Linear(192*6*4, 192),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),
#             nn.Linear(192, 128),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),
#             nn.Linear(128, 2),
#         )
#     def forward(self, x):
#         x = self.features(x)
#         x = torch.flatten(x, 1)
#         x = self.classifier(x)
#         return x

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(6, 48, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(48, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(128, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(192, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(192*4, 192),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(192, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 2),
        )
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class FCN(nn.Module):
    def __init__(self, num_cls=2):
        super(FCN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv1d(in_channels=5, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.conv1d = nn.Conv1d(in_channels=32, out_channels=num_cls, kernel_size=1, stride=1, padding=0)
        self.avg_pool1d = nn.AdaptiveAvgPool1d(1)


    def forward(self, x):
        x = self.features(x)
        x = self.conv1d(x)
        x = self.avg_pool1d(x)
        x = torch.squeeze(x)

        return x