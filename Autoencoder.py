import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.autograd import Function
import torch.nn as nn
import math

zsize = 48
batch_size = 16
iterations = 500
learningRate = 0.0001


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)

        # out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        # out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


###############################################################


###############################################################
class Encoder(nn.Module):
    def __init__(self, block, layers, num_classes=23):
        self.inplanes = 64
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # , return_indices = True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, 1000)
        # self.fc = nn.Linear(num_classes,16)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                # nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        # x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


encoder = Encoder(Bottleneck, [3, 4, 6, 3])
# encoder.load_state_dict(
#     torch.load("/home/deepkliv/Downloads/resnet50-19c8e357.pth")
# )  # ,map_location=lambda storage, loc: storage.cuda(1)),strict=False)
# loaded_weights = torch.load('/home/siplab/Saket/resnet18-5c106cde.pth')
# print encoder.layer1[1].conv1.weight.data[0][0]
encoder.fc = nn.Linear(2048, 48)
# for param in encoder.parameters():
#    param.requires_grad = False
# encoder = encoder.cuda()
# y = torch.rand(1, 3, 224, 224)
# x = torch.rand(1, 128)
# x = Variable(x.cuda())
# print decoder(x)
# y=Variable(y.cuda())
# print("\n")
# encoder(y)
# print encoder(y)
##########################################################################


class Binary(Function):
    @staticmethod
    def forward(ctx, input):
        return F.relu(Variable(input.sign())).data

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


binary = Binary()

##########################################################################
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dfc3 = nn.Linear(zsize, 4096)
        self.bn3 = nn.BatchNorm2d(4096)
        self.dfc2 = nn.Linear(4096, 4096)
        self.bn2 = nn.BatchNorm2d(4096)
        self.dfc1 = nn.Linear(4096, 256 * 6 * 6)
        self.bn1 = nn.BatchNorm2d(256 * 6 * 6)
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.dconv5 = nn.ConvTranspose2d(256, 256, 3, padding=0)
        self.dconv4 = nn.ConvTranspose2d(256, 384, 3, padding=1)
        self.dconv3 = nn.ConvTranspose2d(384, 192, 3, padding=1)
        self.dconv2 = nn.ConvTranspose2d(192, 64, 5, padding=2)
        self.dconv1 = nn.ConvTranspose2d(64, 3, 12, stride=4, padding=4)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # ,i1,i2,i3):
        # print("x.shape dec", x.shape)

        # x = x.view(x.size(0), -1)

        # print("x.shape dec", x.shape)
        x = self.dfc3(x)
        x = self.relu(x)
        # x = self.relu(self.bn3(x.transpose(3, 1)))

        x = self.dfc2(x)
        # x = self.relu(self.bn2(x.transpose(3, 1)))
        # x = self.relu(x)

        x = self.dfc1(x)
        # x = self.relu(self.bn1(x.transpose(3, 1)))
        x = self.relu(x)

        # print(x.shape)
        x = x.view(-1, 256, 6, 6)
        # print (x.size())
        x = self.upsample1(x)
        # print x.size()
        x = self.dconv5(x)
        # print x.size()
        x = self.relu(x)
        # print x.size()
        x = self.relu(self.dconv4(x))
        # print x.size()
        x = self.relu(self.dconv3(x))
        # print x.size()
        x = self.upsample1(x)
        # print x.size()
        x = self.dconv2(x)
        # print x.size()
        x = self.relu(x)
        x = self.upsample1(x)
        # print x.size()
        x = self.dconv1(x)
        # print x.size()
        x = self.sigmoid(x)
        # print x
        return x


##########################################
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = encoder
        self.binary = Binary()
        self.decoder = Decoder()

    def forward(self, x):
        # x=Encoder(x)
        # print(x.shape)
        x = self.encoder(x)
        # print(x.shape)
        # x = binary.apply(x)
        # print(x.shape)

        # print(x.shape)
        # print x
        # x,i2,i1 = self.binary(x)
        # x=Variable(x)
        x = self.decoder(x)
        return x


# print Autoencoder()
