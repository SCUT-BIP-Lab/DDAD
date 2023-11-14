# -*- coding: utf-8 -*-
import torch

import numpy as np
from config import config
from det_oprs.bbox_opr import box_overlap_opr, bbox_transform_opr, box_overlap_ignore_opr

@torch.no_grad()
def fpn_roi_target(rpn_rois, im_info, gt_boxes, top_k=1):
    return_rois = []
    return_labels = []
    return_bbox_targets = []
    # get per image proposals and gt_boxes
    for bid in range(config.train_batch_per_gpu):   #train_batch_per_gpu=2
        gt_boxes_perimg = gt_boxes[bid, :int(im_info[bid, 5]), :]
        #print("-=-=-=stop here -=-=-=-", gt_boxes)
        # print("-=-=-=stop there -=-=-=-",gt_boxes_perimg)   #每张图片tag为1的gt_boxes
        #type_as 进行数据转换
        batch_inds = torch.ones((gt_boxes_perimg.shape[0], 1)).type_as(gt_boxes_perimg) * bid
        gt_rois = torch.cat([batch_inds, gt_boxes_perimg[:, :4]], axis=1)
        batch_roi_inds = torch.nonzero(rpn_rois[:, 0] == bid, as_tuple=False).flatten()
        all_rois = torch.cat([rpn_rois[batch_roi_inds], gt_rois], axis=0)
        overlaps_normal, overlaps_ignore = box_overlap_ignore_opr(
                all_rois[:, 1:5], gt_boxes_perimg)
        overlaps_normal, overlaps_normal_indices = overlaps_normal.sort(descending=True, dim=1)
        overlaps_ignore, overlaps_ignore_indices = overlaps_ignore.sort(descending=True, dim=1)
        # gt max and indices, ignore max and indices
        # 这个flatten有什么用吗，将每个预测的框选出前两个与GT的IoU最大的GT索引，但是下面的flatten是什么目的呢？
        max_overlaps_normal = overlaps_normal[:, :top_k].flatten()
        #print("-=-=-=-max_overlaps_normal-=-=-=-=-", max_overlaps_normal)
        gt_assignment_normal = overlaps_normal_indices[:, :top_k].flatten()
        #print("-=-=-=-gt_assignment_normal-=-=-=-=-", gt_assignment_normal)
        max_overlaps_ignore = overlaps_ignore[:, :top_k].flatten()
        gt_assignment_ignore = overlaps_ignore_indices[:, :top_k].flatten()
        # cons masks
        ignore_assign_mask = (max_overlaps_normal < config.fg_threshold) * (
                max_overlaps_ignore > max_overlaps_normal)
        #print("-=-=-=-ignore_assign_mask-=-=-=-=-", ignore_assign_mask)
        #上一行ignore和normal都不是对应的，这个大于用来做ignore的mask，会不会有问题呢？
        max_overlaps = max_overlaps_normal * ~ignore_assign_mask + \
                max_overlaps_ignore * ignore_assign_mask
        gt_assignment = gt_assignment_normal * ~ignore_assign_mask + \
                gt_assignment_ignore * ignore_assign_mask
        labels = gt_boxes_perimg[gt_assignment, 4]
        fg_mask = (max_overlaps >= config.fg_threshold) * (labels != config.ignore_label)
        bg_mask = (max_overlaps < config.bg_threshold_high) * (
                max_overlaps >= config.bg_threshold_low)
        fg_mask = fg_mask.reshape(-1, top_k)   #反flatten操作，得到多行2列
        bg_mask = bg_mask.reshape(-1, top_k)
        pos_max = config.num_rois * config.fg_ratio    #512*0.5  这是什么意思， 正样本最多多少个？  yes
        fg_inds_mask = subsample_masks(fg_mask[:, 0], pos_max, True)    #这边又是只选了第0列，那是不是就相当于没有top_k=2的说法了？
        neg_max = config.num_rois - fg_inds_mask.sum()
        bg_inds_mask = subsample_masks(bg_mask[:, 0], neg_max, True)
        # print("-=-=labels-=-=-", labels)
        # print("-=-=fg_mask-=-=-", fg_mask)
        # print("-=-=fg_mask-=-=-", fg_mask.flatten)
        labels = labels * fg_mask.flatten()
        keep_mask = fg_inds_mask + bg_inds_mask
        # labels
        labels = labels.reshape(-1, top_k)[keep_mask]
        gt_assignment = gt_assignment.reshape(-1, top_k)[keep_mask].flatten()
        target_boxes = gt_boxes_perimg[gt_assignment, :4]
        rois = all_rois[keep_mask]
        target_rois = rois.repeat(1, top_k).reshape(-1, all_rois.shape[-1])
        bbox_targets = bbox_transform_opr(target_rois[:, 1:5], target_boxes)
        if config.rcnn_bbox_normalize_targets:
            std_opr = torch.tensor(config.bbox_normalize_stds[None, :]).type_as(bbox_targets)
            mean_opr = torch.tensor(config.bbox_normalize_means[None, :]).type_as(bbox_targets)
            minus_opr = mean_opr / std_opr
            bbox_targets = bbox_targets / std_opr - minus_opr
        bbox_targets = bbox_targets.reshape(-1, top_k * 4)
        return_rois.append(rois)
        return_labels.append(labels)
        return_bbox_targets.append(bbox_targets)
    if config.train_batch_per_gpu == 1:
        return rois, labels, bbox_targets
    else:
        return_rois = torch.cat(return_rois, axis=0)
        return_labels = torch.cat(return_labels, axis=0)
        return_bbox_targets = torch.cat(return_bbox_targets, axis=0)
        return return_rois, return_labels, return_bbox_targets

def subsample_masks(masks, num_samples, sample_value):
    positive = torch.nonzero(masks.eq(sample_value), as_tuple=False).squeeze(1)
    num_mask = len(positive)
    num_samples = int(num_samples)
    num_final_samples = min(num_mask, num_samples)
    num_final_negative = num_mask - num_final_samples
    # print("-=-=num_final_negative-=-=-",num_final_negative)
    #这边有点奇怪，对于一般少于250个正样本的情况，这个值就变成0，同时下一行又有一个[:0]的作用是？
    perm = torch.randperm(num_mask, device=masks.device)[:num_final_negative]    #随机打乱顺序
    # print("-=-=perm-=-=-", perm)
    negative = positive[perm]
    # print("-=-=negative-=-=-", negative)
    masks[negative] = not sample_value
    # print("-=-=masks-=-=-", masks)
    return masks

