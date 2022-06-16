import torch
import torch.nn as nn
from torch.nn import functional as F

class MLP(nn.Module):
    def __init__(self, input_size=28, hidden_units=512, num_classes=10, dropout=0):
        super(MLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size * input_size, hidden_units),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(hidden_units, num_classes),
        )

    def forward(self, image):
        image = image.view(image.size(0), -1)
        output = self.model(image)
        return output


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1)
        self.fc = nn.Linear(8 * 7 * 7, 10)
        self.resize = nn.Upsample(28, mode="bicubic")

    def forward(self, x):
        x = self.resize(x)
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 8 * 7 * 7)
        return self.fc(x)


class small_cnn_imagenet(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, dropout=0):
        super(small_cnn_imagenet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, 3)
        self.conv1_drop = nn.Dropout2d(p=dropout)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.conv2_drop = nn.Dropout2d(p=dropout)
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.conv3_drop = nn.Dropout2d(p=dropout)
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.conv3_drop = nn.Dropout2d(p=dropout)
        self.conv4 = nn.Conv2d(256, 256, 3)
        self.conv4_drop = nn.Dropout2d(p=dropout)
        self.fc1 = nn.Linear(256 * 12 * 12, 1024)
        self.fc1_drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(1024, 512)
        self.fc2_drop = nn.Dropout(dropout)
        self.fc3 = nn.Linear(512, 256)
        self.fc3_drop = nn.Dropout(dropout)
        self.fc4 = nn.Linear(256, num_classes)

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = F.relu(self.pool(self.conv1_drop(self.conv1(x))))
        x = F.relu(self.pool(self.conv2_drop(self.conv2(x))))
        x = F.relu(self.pool(self.conv3_drop(self.conv3(x))))
        x = F.relu(self.pool(self.conv4_drop(self.conv4(x))))
        x = x.view(-1, 256 * 12 * 12)
        x = F.relu(self.fc1_drop(self.fc1(x)))
        x = F.relu(self.fc2_drop(self.fc2(x)))
        x = F.relu(self.fc3_drop(self.fc3(x)))
        x = self.fc4(x)
        return x
        
class small_cnn(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, dropout=0):
        super(small_cnn, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, 3)
        self.conv1_drop = nn.Dropout2d(p=dropout)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.conv2_drop = nn.Dropout2d(p=dropout)
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.conv3_drop = nn.Dropout2d(p=dropout)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc1_drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 256)
        self.fc2_drop = nn.Dropout(dropout)
        self.fc3 = nn.Linear(256, num_classes)

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = F.relu(self.pool(self.conv1_drop(self.conv1(x))))
        x = F.relu(self.pool(self.conv2_drop(self.conv2(x))))
        x = F.relu(self.pool(self.conv3_drop(self.conv3(x))))

        x = x.view(-1, 64 * 4 * 4)

        x = F.relu(self.fc1_drop(self.fc1(x)))
        x = F.relu(self.fc2_drop(self.fc2(x)))
        x = self.fc3(x)
        return x


class wide_cnn(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, dropout=0):
        super(wide_cnn, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 128, 3)
        self.conv1_drop = nn.Dropout2d(p=dropout)
        self.conv2 = nn.Conv2d(128, 256, 3)
        self.conv2_drop = nn.Dropout2d(p=dropout)
        self.conv3 = nn.Conv2d(256, 256, 3)
        self.conv3_drop = nn.Dropout2d(p=dropout)
        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        self.fc1_drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, 512)
        self.fc2_drop = nn.Dropout(dropout)
        self.fc3 = nn.Linear(512, num_classes)

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = F.relu(self.pool(self.conv1_drop(self.conv1(x))))
        x = F.relu(self.pool(self.conv2_drop(self.conv2(x))))
        x = F.relu(self.pool(self.conv3_drop(self.conv3(x))))

        x = x.view(-1, 64 * 4 * 4)

        x = F.relu(self.fc1_drop(self.fc1(x)))
        x = F.relu(self.fc2_drop(self.fc2(x)))
        x = self.fc3(x)
        return x

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)
        self.relu5 = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)
        y = self.relu5(y)
        return y


class LeNet5_40x40(nn.Module):
    def __init__(self):
        super(LeNet5_40x40, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(784, 784)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(784, 512)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(512, 10)
        self.relu5 = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)
        y = self.relu5(y)
        return y
