import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import voc, coco
import os
import mobilenetv2_fpn as mobilenetv2
import numpy 


class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, head, num_classes):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes

        # self.cfg = (coco, voc)[num_classes == 21]
        self.cfg = (coco, voc)[num_classes == 2]

        self.priorbox = PriorBox(self.cfg)

        #self.priors = Variable(self.priorbox.forward(), volatile=True)
        with torch.no_grad():
            self.priors = Variable(self.priorbox.forward())
            # print('self.priors.size():', self.priors.size())
        self.size = size

        # SSD network
        self.mobilenet = base

        # self.neck = neck
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)

        self.fpn = PyramidFeatures(128, 320)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu
        x = self.mobilenet.conv1(x)
        x = self.mobilenet.bn1(x)
        x = self.mobilenet.activation(x)

        for i in self.mobilenet.bottlenecks[:5]:
            # x = self.mobilenet[i](x)
            x = i(x)

        #s = self.L2Norm(x)
        # print('x :5 shape: ', x.shape)      # torch.Size([1, 128, 19, 19])
        sources.append(x)

        # apply vgg up to fc7
        for i in self.mobilenet.bottlenecks[5:]:
            # x = self.mobilenet[i](x)
            x = i(x)

        # print('x 5: shape: ', x.shape)      # torch.Size([1, 320, 10, 10])
        sources.append(x)
        x = self.mobilenet.conv_last(x)
        x = self.mobilenet.bn_last(x)
        x = self.mobilenet.activation(x)

        # sources.append(x)

        features = self.fpn(sources)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = v(x)
            if k % 2 == 1:
                features.append(x)
    

        # apply multibox head to source layers
        for (x, l, c) in zip(features, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())


        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
    

        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf preds
                # torch.sigmoid(conf.view(conf.size(0), -1,       ## sigmoid prediction
                #              self.num_classes)),
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                # conf.view(conf.size(0), -1, 1),         ## sigmoid prediction
                self.priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location='cuda:0'))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


def conv_dw(inp, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp,bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU6(inplace=True),
    )

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1,bias=False),
        nn.BatchNorm2d(oup),
        # nn.ReLU6(inplace=True)
        nn.ReLU(inplace=True)
    )

def conv1_bn(inp, oup ,stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0,bias=False),
        nn.BatchNorm2d(oup),
        # nn.ReLU6(inplace=True),
        nn.ReLU(inplace=True),
    )


class PyramidFeatures(nn.Module):
    # def __init__(self, C3_size, C4_size, C5_size, feature_size=128):
    def __init__(self, C4_size, C5_size, feature_size=128):
        super(PyramidFeatures, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        # self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_upsampled = nn.ConvTranspose2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.ConvTranspose2d(feature_size, feature_size, kernel_size=2, stride=2)
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # # add P4 elementwise to C3
        # self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        # self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs):
        # C3, C4, C5 = inputs
        C4, C5 = inputs
        # print('C4:', type(C4), C4.size())   # torch.Size([1, 128, 19, 19])
        # print('C5:', type(C5), C5.size())   # ([1, 320, 10, 10])
        P5_x = self.P5_1(C5)
        # print('P5_x:', P5_x.size())
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        # print('P4_x:', P4_x.size())
        P4_x = P5_upsampled_x + P4_x

        # P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)
        # print('P4_x:', P4_x.size())

        # P3_x = self.P3_1(C3)
        # P3_x = P3_x + P4_upsampled_x

        # P3_x = self.P3_2(P3_x)

        # return [P3_x, P4_x, P5_x]
        return [P4_x, P5_x]


def add_extras(i):      ## 在此处加入fpn
    # Extra layers added to VGG for feature scaling
    layers = []
    # tensor_layers = []

    #conv14
    layers += [conv1_bn(i,256,1)]
    layers += [conv_bn(256,512,2)]

    #conv15
    layers += [conv1_bn(512,128,1)]
    layers += [conv_bn(128,256,2)]

    #con16
    layers += [conv1_bn(256,128,1)]
    layers += [conv_bn(128,256,2)]

    #conv17
    layers += [conv1_bn(256,64,1)]
    layers += [conv_bn(64,128,2)]
    # print('layers:', len(layers), type(layers))


    return layers


def multibox(mobilenet, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []

    ### may be have bug ###
    #mobilenetv2_source = [5, -1]
    extras_source = [1,3,5,7]

    # loc_layers += [nn.Conv2d(96, 4 * 4, kernel_size=1)]
    # conf_layers += [nn.Conv2d(96, 4 * num_classes, kernel_size=1)]

    loc_layers += [nn.Conv2d(128, 4 * 4, kernel_size=1)]
    conf_layers += [nn.Conv2d(128, 4 * num_classes, kernel_size=1)]

    # loc_layers += [nn.Conv2d(1280, 6 * 4, kernel_size=1)]
    # conf_layers += [nn.Conv2d(1280, 6 * num_classes, kernel_size=1)]
    loc_layers += [nn.Conv2d(128, 6 * 4, kernel_size=1)]
    conf_layers += [nn.Conv2d(128, 6 * num_classes, kernel_size=1)]

    # for k, v in enumerate(extra_layers[1::2], 2):
    for k, v in enumerate(extras_source):
        k += 2
        loc_layers += [nn.Conv2d(extra_layers[v][0].out_channels,
                                 cfg[k] * 4, kernel_size=1)]
        conf_layers += [nn.Conv2d(extra_layers[v][0].out_channels,
                                  cfg[k] * num_classes, kernel_size=1)]
    return mobilenet, extra_layers, (loc_layers, conf_layers)

extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256],
    '512': [],
}

mbox = {
    '300':[4, 6, 6, 6, 6, 6],
    '512': [],
}


def build_ssd(phase, size=300, num_classes=2):

    # add, no use
    size = 300

    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 300:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 (size=300) is supported!")
        return

    # base_, extras_, head_ = multibox(mobilenetv2.MobileNet2(scale=1.0), add_extras(1280),mbox[str(size)], num_classes)
    base_, extras_, head_ = multibox(mobilenetv2.MobileNet2(scale=1.0), add_extras(128),mbox[str(size)], num_classes)
    
    return SSD(phase, size, base_, extras_, head_, num_classes)


if __name__ =="__main__":
    torch.backends.cudnn.enabled = False
    ssd = build_ssd("train")
    x = torch.zeros((32, 96, 19, 19))
    x = ssd.loc[0](x)
    print(x.size())
    x = torch.zeros((32, 1280, 10, 10))
    x = ssd.loc[1](x)
    print(x.size())
    x = torch.zeros((32, 512, 5, 5))
    x = ssd.loc[2](x)
    print(x.size())
    x = torch.zeros((32, 256, 3, 3))
    x = ssd.loc[3](x)
    print(x.size())
    x = torch.zeros((32, 256, 2, 2))
    x = ssd.loc[4](x)
    print(x.size())
    x = torch.zeros((32, 128, 1, 1))
    x = ssd.loc[5](x)
    print(x.size())

