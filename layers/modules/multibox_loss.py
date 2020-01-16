# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data import coco as cfg
from ..box_utils import match, log_sum_exp, match_ious, bbox_overlaps_iou, bbox_overlaps_giou, bbox_overlaps_diou, bbox_overlaps_ciou, decode
import focal_loss
import sigmoid_focal_loss
import ghm_loss
import ghm_loss_official
import diou_loss
import numpy as np
import data.config as cfgg


class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,
                 use_gpu=True):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = cfg['variance']
        # print('self.variance:', self.variance)

    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            targets (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        loc_data, conf_data, priors = predictions
        num = loc_data.size(0)
        priors = priors[:loc_data.size(1), :]
        num_priors = (priors.size(0))
        num_classes = self.num_classes

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)

        # conf_t = torch.zeros(num,num_priors).long()

        for idx in range(num):
            target = targets[idx]
            truths = target[:, :-1].data
            labels = target[:, -1].data
            defaults = priors.data
            if cfgg.USE_DIOU:
                match_ious(self.threshold, truths, defaults, self.variance, labels,
                    loc_t, conf_t, idx)
            else:
                match(self.threshold, truths, defaults, self.variance, labels,
                    loc_t, conf_t, idx)

        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)

        pos = conf_t > 0
        num_pos = pos.sum(dim=1, keepdim=True)

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)

        if cfgg.USE_DIOU:
            giou_priors = priors.data.unsqueeze(0).expand_as(loc_data)
            compute_l_loss = diou_loss.IouLoss(pred_mode='Center',size_sum=True,variances=self.variance, losstype='Diou')
            loss_l = compute_l_loss(loc_p,loc_t,giou_priors[pos_idx].view(-1, 4))
        else:
            loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        # loss_c[pos] = 0  # filter out pos boxes for now
        # loss_c = loss_c.view(num, -1)

        # Hard Negative Mining
        loss_c = loss_c.view(num, -1)
        loss_c[pos] = 0

        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)

        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        # print('conf_p size:', conf_p.size())
        # print('targets_weighted size:', targets_weighted.size())
        # label_weight = torch.zeros_like(priors)
        # if len(pos_idx) > 0:
        #     label_weight[pos_idx, :] = 1.0
        label_weight = torch.ones_like(conf_p)

        if cfgg.USE_FL:
            # alpha = np.array([[0.25], [0.75], [0.75], [0.75], [0.75],
            #                   [0.75], [0.75], [0.75], [0.75], [0.75],
            #                   [0.75], [0.75], [0.75], [0.75], [0.75],
            #                   [0.75], [0.75], [0.75], [0.75], [0.75], [0.75]])
            ## softmax focal loss
            alpha = np.array([[0.25], [0.75]])
            alpha = torch.Tensor(alpha)
            compute_c_loss = focal_loss.FocalLoss(alpha=alpha, gamma=2, class_num=num_classes, size_average=False)
            loss_c = compute_c_loss(conf_p, targets_weighted)

            ## sigmoid focal loss
            # focalloss = sigmoid_focal_loss.SigmoidFocalLoss(gamma=2, alpha=0.25)
            # loss_c = focalloss(conf_p, targets_weighted)
        elif cfgg.USE_GHM:
            ## unofficial ghm loss
            # compute_c_loss = ghm_loss.GHMC_Loss(bins=10, alpha=0.25)
            # loss_c = compute_c_loss(conf_p, targets_weighted)
            # compute_l_loss = ghm_loss.GHMR_Loss(bins=10, alpha=0.25)
            # loss_l = compute_l_loss(conf_p, targets_weighted)

            ## official ghm loss
            compute_c_loss = ghm_loss_official.GHMC()
            loss_c = compute_c_loss(conf_p, targets_weighted, label_weight)
            compute_l_loss = ghm_loss_official.GHMR()
            loss_l = compute_l_loss(loc_p, loc_t, label_weight)       
        else:
            loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

        N = num_pos.data.sum()

        loss_l /= N
        loss_c /= N

        # print("N",N,"\t","loss_l",loss_l,"\t","loss_c",loss_c)

        return loss_l, loss_c
