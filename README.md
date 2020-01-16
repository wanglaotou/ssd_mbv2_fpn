# SSD_mobilenetv2_fpn

 this repo is forked from https://github.com/amdegroot/ssd.pytorch. Implement by pytorch.

add functions:
1. implement mobielentv2 for ssd.
2. add focal loss. (need adjuct super-parameters).
3. add fpn module. (change the channel, so you should train your datasets from scratch.)
4. add ssd_test_mb_folder.py demo for image detection.
5. add ssd_test_mb_video.py demo for video detection.
6. add eval/voc_eval.py for map.

train ssd_mbv2_fpn:
python train_mobilenetv2_fpn.py

train ssd_mbv2:
python train_mobilenetv2.py

test:
python ssd_test_mb_folder.py

evaluation:
python eval/voc_eval.py

notice:
I add one fpn layer in mobilenetv2, and change the channel number, so there is some different between the origin mobilenetv2, but in my experiments, i trained my own datasets from scratch and it gets better results.
but if you want train ssd_mobilenetv2, you can use the pretrained model in the weights.
