import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import cv2
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

from ssd_mobilenetv2_fpn import build_ssd   # train with mobilenet backbone
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

net = build_ssd('test', 300, 2)    # initialize SSD
net.load_state_dict(torch.load('weights_zhongdong/ssd_mobilenetv2_fpn_20200107/mobilenetv2_290000.pth'))  # test mobilenet backbone
net.eval()

# image = cv2.imread('./data/example.jpg', cv2.IMREAD_COLOR)  # uncomment if dataset not downloaded
# %matplotlib inline
from matplotlib import pyplot as plt
from data import VOCDetection, VOC_ROOT, VOCAnnotationTransform
# here we specify year (07 or 12) and dataset ('test', 'val', 'train') 
root_test = '/media/mario/新加卷/DataSets/ALPR/zhongdong'
save_root = '/home/mario/Projects/SSD/SSD_mobilenetv2'
img_root = os.path.join(root_test,'JPEGImages')
ftr = open('eval/plate_mb_result_zhongdong_fpn_29w_thre045.txt','w')
test_txtpath = os.path.join(root_test,'ImageSets/Main/test.txt')
test_imgpath = os.path.join(save_root,'eval/plate_mb_result_zhongdong_fpn_29w_thre045')
if not os.path.exists(test_imgpath):
    os.mkdir(test_imgpath)
ft = open(test_txtpath,'r')
ftlines = ft.readlines()
for fline in ftlines:
    imgfor = fline.strip()
    imgname = imgfor + '.jpg'
    imgpath = os.path.join(img_root,imgname)
    save_imgpath = os.path.join(test_imgpath,imgname)
    print(imgpath)
    image = cv2.imread(imgpath)
    height, width, channel = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # print('rgb_image shape:', rgb_image.shape)

    x = cv2.resize(image, (300, 300)).astype(np.float32)
    x -= (104.0, 117.0, 123.0)
    x = x.astype(np.float32)
    x = x[:, :, ::-1].copy()
    # print('x size:', x.shape)
    
    x = torch.from_numpy(x).permute(2, 0, 1)

    xx = Variable(x.unsqueeze(0))     # wrap tensor in Variable
    # print('xx size:', xx.size())

    if torch.cuda.is_available():
        xx = xx.to(device)

    y = net(xx)

    from data import VOC_CLASSES as labels
    top_k=10

    detections = y.data
    # scale each detection back up to the image64
    scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)

    for i in range(detections.size(1)):
        j = 0
        # print('score:', detections[0,i,j,0])
        while detections[0,i,j,0] >= 0.45:
            score = detections[0,i,j,0]
            # ftr.write(imgfor+' '+str(np.round(score.cpu().numpy(),3)))
            label_name = labels[i-1]
            display_txt = '%s: %.2f'%(label_name, score)
            if(score>=0.45):
                # print('display_txt:', display_txt)
                pt = (detections[0,i,j,1:]*scale).cpu().numpy()
                coords = (pt[0], pt[1]), pt[2]-pt[0]+1, pt[3]-pt[1]+1
                # print('coords:', coords)
                ftr.write(imgfor+' '+str(np.round(score.cpu().numpy(),3))+' '+str(int(pt[0]))+' '+str(int(pt[1]))+' '+str(int(pt[2]))+' '+str(int(pt[3]))+'\n')
                cv2.putText(image, str(np.round(score.cpu().numpy(),3)),(pt[0], pt[1]), cv2.FONT_HERSHEY_PLAIN, 2, color=(255,0,255))
                cv2.rectangle(image, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (0, 0, 255), 2)
            j+=1
        cv2.imwrite(save_imgpath,image)
    # cv2.imshow('image', image)
    # cv2.waitKey(0)
        
