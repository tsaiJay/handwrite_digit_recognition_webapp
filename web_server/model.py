import torch
import torch.nn as nn


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.pool2 = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(4*4*16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.relu = nn.ReLU()

    def forward(self, x):  # x: input
        out = self.conv1(x)
        out = self.relu(out)
        out = self.pool1(out)
        # print('layer1', out.shape)

        out = self.conv2(out)
        out = self.relu(out)
        out = self.pool2(out)
        # print('layer2', out.shape)

        out = out.flatten(1)

        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out


if __name__ == "__main__":
    fake = torch.randn(16, 1, 28, 28)
    
    net = LeNet()

    print('input size', fake.shape)
    out = net(fake)

    print('output size', out.shape)