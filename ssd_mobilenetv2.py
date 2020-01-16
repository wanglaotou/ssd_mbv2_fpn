import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import voc, coco
import os
import mobilenetv2 as mobilenetv2
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
        #self.vgg = nn.ModuleList(base)
        self.mobilenet = base

        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)
            #self.detect = Detect(num_classes, 0, 20, 0.01, 0.45)
            #self.detect = Detect(num_classes, 0, 200, 0.1, 0.45)

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

        # for i in self.mobilenet.bottlenecks[:2]:
        #     x = i(x)

        # print('x bottleneck_2:', x.shape)
        # pytorch_bottleneck_2 = open('pytorch_bottleneck_2.txt','w')
        # x_md2_np = x.cpu().detach().numpy()
        # for i in range(1):
        #     for j in range(24):
        #         for m in range(75):
        #             for n in range(75):
        #                 pytorch_bottleneck_2.write(str(x_md2_np[i][j][m][n])+'\n')

        for i in self.mobilenet.bottlenecks[:5]:
            x = i(x)

        #s = self.L2Norm(x)
        sources.append(x)

        # print('x bottleneck_middle:', x.shape)
        # pytorch_bottleneck_middle = open('pytorch_bottleneck_middle.txt','w')
        # x_md_np = x.cpu().detach().numpy()
        # for i in range(1):
        #     for j in range(96):
        #         for m in range(19):
        #             for n in range(19):
        #                 pytorch_bottleneck_middle.write(str(x_md_np[i][j][m][n])+'\n')

        # apply vgg up to fc7

        for i in self.mobilenet.bottlenecks[5:]:
            x = i(x)
        x = self.mobilenet.conv_last(x)
        x = self.mobilenet.bn_last(x)
        x = self.mobilenet.activation(x)

        sources.append(x)
        # print('x bottleneck:', x.shape)
        # pytorch_bottleneck = open('pytorch_extra_before.txt','w')
        # x_np = x.cpu().detach().numpy()
        # for i in range(x.shape[0]):
        #     for j in range(x.shape[1]):
        #         for m in range(x.shape[2]):
        #             for n in range(x.shape[3]):
        #                 pytorch_bottleneck.write(str(x_np[i][j][m][n])+'\n')

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            #print(x.size())
            #print(v(x).size())
            # x = F.relu(v(x), inplace=True)
            x = v(x)
            if k % 2 == 1:
                sources.append(x)
        
        # print('x bottleneck:', x.shape)
        # pytorch_bottleneck = open('pytorch_extra_after.txt','w')        ##  ->  ok
        # x_np = x.cpu().detach().numpy()
        # for i in range(x.shape[0]):
        #     for j in range(x.shape[1]):
        #         for m in range(x.shape[2]):
        #             for n in range(x.shape[3]):
        #                 pytorch_bottleneck.write(str(x_np[i][j][m][n])+'\n')

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        # for (x, l, c) in zip(sources, self.loc, self.conf):
        #     loc.append(l(x))
        #     conf.append(c(x))
        
        # print('x bottleneck:', x.shape)
        # pytorch_bottleneck = open('pytorch_loc_after.txt','w')        ##  ->  ok
        # x_np = x.cpu().detach().numpy()
        # for i in range(x.shape[0]):
        #     for j in range(x.shape[1]):
        #         for m in range(x.shape[2]):
        #             for n in range(x.shape[3]):
        #                 pytorch_bottleneck.write(str(x_np[i][j][m][n])+'\n')

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        # print('loc:',loc.shape)
        # print('conf:', conf.shape)
        # pytorch_conf = open('pytorch_conf.txt','w')
        # pytorch_loc = open('pytorch_loc.txt','w')
        # loc_np = loc.cpu().detach().numpy()
        # conf_np = conf.cpu().detach().numpy()
        # for i in range(2278):
        #     pytorch_conf.write(str(conf_np[0][i:i+2])+'\n')
        #     i+=1
        # for i in range(2278): 
        #     pytorch_loc.write(str(loc_np[0][i:i+4])+'\n')
        #     i+=3

        if self.phase == "test":
            # print('x.data:', x.data.size())
            # print('self.priors:',self.priors.size())
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


def add_extras(i):
    # Extra layers added to VGG for feature scaling
    layers = []

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
    print('layers:', type(layers), len(layers))
    return layers


def multibox(mobilenet, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []

    ### may be have bug ###
    #mobilenetv2_source = [5, -1]
    extras_source = [1,3,5,7]

    loc_layers += [nn.Conv2d(96, 4 * 4, kernel_size=1)]
    conf_layers += [nn.Conv2d(96, 4 * num_classes, kernel_size=1)]

    loc_layers += [nn.Conv2d(1280, 6 * 4, kernel_size=1)]
    conf_layers += [nn.Conv2d(1280, 6 * num_classes, kernel_size=1)]

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
    # base_, extras_, head_ = multibox(vgg(base[str(size)], 3),
    #                                  add_extras(extras[str(size)], 1024),
    #                                  mbox[str(size)], num_classes)

    base_, extras_, head_ = multibox(mobilenetv2.MobileNet2(scale=1.0), add_extras(1280),mbox[str(size)], num_classes)

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


    # x = torch.zeros((32, 1280, 10, 10))
    #
    # for i in ssd.extras:
    #     x = i(x)
    #     print(x.size())

    # x = ssd.mobilenet.conv1(x)
    # print(x.size())
    # x = ssd.mobilenet.bn1(x)
    # print(x.size())
    # x = ssd.mobilenet.relu(x)
    # print(x.size())
    # x = ssd.mobilenet.layer1(x)
    # print(x.size())
    # x = ssd.mobilenet.layer2(x)
    # print(x.size())
    # x = ssd.mobilenet.layer3(x)
    # print(x.size())
    # x = ssd.mobilenet.layer4(x)
    # print(x.size())
    # x = ssd.mobilenet.layer5(x)
    # print(x.size())
    # x = ssd.mobilenet.layer6(x)
    # print(x.size())
    # x = ssd.mobilenet.layer7(x)
    # print(x.size())
    # x = ssd.mobilenet.layer8(x)
    # print(x.size())
    # x = ssd.mobilenet.conv9(x)
    # print(x.size())