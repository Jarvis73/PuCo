import torch.nn as nn


class FCDiscriminator(nn.Module):
    """
    inplanes, planes. Patch-gan
    """

    def __init__(self, inplanes, planes = 64):
        super(FCDiscriminator, self).__init__()
        self.name_ = self.__class__.__name__
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(planes, planes * 2, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(planes * 2, planes * 4, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(planes * 4, planes * 8, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.classifier = nn.Conv2d(planes * 8, 1, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        return x
