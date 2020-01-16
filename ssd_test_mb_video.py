#coding:utf-8
import os
import sys, random
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

from ssd_mobilenetv2 import build_ssd   # train with mobilenet backbone
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

net = build_ssd('test', 300, 2)    # initialize SSD
net.load_state_dict(torch.load('weights/ssd_mobilenetv2_300_focal_relu/mobilenetv2_final.pth'))  # test mobilenet backbone
net.eval()

video_test_path = '/media/mario/新加卷/DataSets/videosrc'
img_paths = []
img_paths = [el for el in os.listdir(video_test_path)]
random.shuffle(img_paths)
num = len(img_paths)
print("%d videos in total." % num)
save_dir = '/media/mario/新加卷/DataSets/result_video'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)


for annotation in img_paths:
    videoname = annotation
    videopath = os.path.join(video_test_path,videoname)
    videofor = videoname.split('.avi')[0]
    save_path = os.path.join(save_dir, videofor)
    pred_txt = os.path.join(save_dir,str(videofor)+'.txt')
    
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    frame_count = 1
    print(videopath)
    cap = cv2.VideoCapture(videopath)
    while True:
        success, image = cap.read()
        print(image.shape)
        if not success:
            break
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
        # scale each detection back up to the image
        scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)
        for i in range(detections.size(1)):
            j = 0
            # print('score:', detections[0,i,j,0])
            while detections[0,i,j,0] >= 0.5:
                score = detections[0,i,j,0]
                # ftr.write(imgfor+' '+str(np.round(score.cpu().numpy(),3)))
                label_name = labels[i-1]
                display_txt = '%s: %.2f'%(label_name, score)
                if(score>=0.5):
                    # print('display_txt:', display_txt)
                    pt = (detections[0,i,j,1:]*scale).cpu().numpy()
                    coords = (pt[0], pt[1]), pt[2]-pt[0]+1, pt[3]-pt[1]+1
                    # print('coords:', coords)
                    cv2.putText(image, str(np.round(score.cpu().numpy(),3)),(pt[0], pt[1]), cv2.FONT_HERSHEY_PLAIN, 2, color=(255,0,255))
                    cv2.rectangle(image, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (0, 0, 255), 2)
                j+=1
            cv2.imshow('image', image)
            cv2.waitKey(1)
            # cv2.imwrite(save_imgpath,image)
  


'''
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
    # scale each detection back up to the image
    scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)

    for i in range(detections.size(1)):
        j = 0
        # print('score:', detections[0,i,j,0])
        while detections[0,i,j,0] >= 0.5:
            score = detections[0,i,j,0]
            # ftr.write(imgfor+' '+str(np.round(score.cpu().numpy(),3)))
            label_name = labels[i-1]
            display_txt = '%s: %.2f'%(label_name, score)
            if(score>=0.5):
                # print('display_txt:', display_txt)
                pt = (detections[0,i,j,1:]*scale).cpu().numpy()
                coords = (pt[0], pt[1]), pt[2]-pt[0]+1, pt[3]-pt[1]+1
                # print('coords:', coords)
                ftr.write(imgfor+' '+str(np.round(score.cpu().numpy(),3))+' '+str(int(pt[0]))+' '+str(int(pt[1]))+' '+str(int(pt[2]))+' '+str(int(pt[3]))+'\n')
                cv2.putText(image, str(np.round(score.cpu().numpy(),3)),(pt[0], pt[1]), cv2.FONT_HERSHEY_PLAIN, 2, color=(255,0,255))
                cv2.rectangle(image, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (0, 0, 255), 2)
            j+=1
        # cv2.imshow('image', image)
        # cv2.waitKey(0)
        cv2.imwrite(save_imgpath,image)
'''