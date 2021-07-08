import torchvision.models as models
import torch.nn as nn

def load_net():
    net = models.resnet18()
    net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    net.fc = nn.Linear(512, 10)
    return net
load_net()
