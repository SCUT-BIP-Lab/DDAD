import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from config import config
from backbone.resnet50 import ResNet50
from backbone.fpn_idad import FPN
from module.rpn import RPN
from layers.pooler import roi_pooler
from det_oprs.bbox_opr import bbox_transform_inv_opr
from det_oprs.fpn_roi_target import fpn_roi_target
from det_oprs.loss_opr import emd_loss_softmax
from det_oprs.utils import get_padded_tensor,get_denisty_mask_padded_tensor



class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet50 = ResNet50(config.backbone_freeze_at, False)
        self.FPN = FPN(self.resnet50, 2, 6, return_base_f  = True)
        self.RPN = RPN(config.rpn_channel)
        self.RCNN = RCNN()
        self.Density = Density(256)

    def forward(self, image,self_mask, im_info,gt_density=None,  all_num=0,gt_boxes=None):
        # print("-=network-=image1---aahudhiuw", image.size())
        image = (image - torch.tensor(config.image_mean[None, :, None, None]).type_as(image)) / (
            torch.tensor(config.image_std[None, :, None, None]).type_as(image))
        # print("-=network-=image---aahudhiuw", image.size())
        image = get_padded_tensor(image, 64)
        # print("-=network-=image---aahudhiuw", image.size())
        # print("-=network-=self_mask---aahudhiuw", self_mask.size())
        # print("-------------------",gt_density.size())
        # print("-------------------", im_info)


        # print("-=network-=gt_density---aahudhiuw", gt_density.size())

        # im_info[]
        if self.training:
            self_mask = get_denisty_mask_padded_tensor(self_mask, [image.shape[-2] // 4, image.shape[-1] // 4])
            # print()
            gt_density = get_denisty_mask_padded_tensor(gt_density, [image.shape[-2] // 4, image.shape[-1] // 4])
            # print("network---------------",self_mask.squeeze().size())
            # np.save("/home/data/CrowdHuman/training_create_npy/" + "1.npy",self_mask.squeeze().cpu().numpy())
            return self._forward_train(image, gt_density, self_mask,im_info,all_num, gt_boxes)
        else:
            return self._forward_test(image, im_info)

    def _forward_train(self, image, gt_density,self_mask, im_info,all_num, gt_boxes):
        loss_dict = {}
        last_f, fpn_fms = self.FPN(image)
        # fpn_fms stride: 64,32,16,8,4, p6->p2
        rpn_rois, loss_dict_rpn = self.RPN(fpn_fms, im_info, gt_boxes)
        rcnn_rois, rcnn_labels, rcnn_bbox_targets = fpn_roi_target(
            rpn_rois, im_info, gt_boxes, top_k=2)
        loss_dict_rcnn = self.RCNN(fpn_fms, rcnn_rois,
                                   rcnn_labels, rcnn_bbox_targets)
        # print("bbbbbbbbbbbbbbbbbb",type(fpn_fms[-1]))
        # print("ccccccccccccccc",fpn_fms[-1].size())
        # print("ddddddddddddddd",gt_density.size())
        loss_dict_density = self.Density(last_f,all_num, target = gt_density,mask = self_mask)
        loss_dict.update(loss_dict_rpn)
        loss_dict.update(loss_dict_rcnn)
        loss_dict.update(loss_dict_density)
        return loss_dict

    def _forward_test(self, image, im_info):
        last_f, fpn_fms = self.FPN(image)

        rpn_rois = self.RPN(fpn_fms, im_info)
        pred_bbox = self.RCNN(fpn_fms, rpn_rois)

        density = self.Density(last_f)
        # density = density*self_mask

        return pred_bbox.cpu().detach(), density.cpu().detach()


class RCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # roi head
        self.fc1 = nn.Linear(256 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 1024)

        for l in [self.fc1, self.fc2]:
            nn.init.kaiming_uniform_(l.weight, a=1)
            nn.init.constant_(l.bias, 0)
        # box predictor
        self.emd_pred_cls_0 = nn.Linear(1024, config.num_classes)
        self.emd_pred_delta_0 = nn.Linear(1024, config.num_classes * 4)
        self.emd_pred_cls_1 = nn.Linear(1024, config.num_classes)
        self.emd_pred_delta_1 = nn.Linear(1024, config.num_classes * 4)
        for l in [self.emd_pred_cls_0, self.emd_pred_cls_1]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)
        for l in [self.emd_pred_delta_0, self.emd_pred_delta_1]:
            nn.init.normal_(l.weight, std=0.001)
            nn.init.constant_(l.bias, 0)

    def forward(self, fpn_fms, rcnn_rois, labels=None, bbox_targets=None):
        # stride: 64,32,16,8,4 -> 4, 8, 16, 32
        fpn_fms = fpn_fms[1:][::-1]
        stride = [4, 8, 16, 32]
        pool_features = roi_pooler(fpn_fms, rcnn_rois, stride, (7, 7), "ROIAlignV2")
        flatten_feature = torch.flatten(pool_features, start_dim=1)
        flatten_feature = F.relu_(self.fc1(flatten_feature))
        flatten_feature = F.relu_(self.fc2(flatten_feature))
        pred_emd_cls_0 = self.emd_pred_cls_0(flatten_feature)
        pred_emd_delta_0 = self.emd_pred_delta_0(flatten_feature)
        pred_emd_cls_1 = self.emd_pred_cls_1(flatten_feature)
        pred_emd_delta_1 = self.emd_pred_delta_1(flatten_feature)
        if self.training:
            loss0 = emd_loss_softmax(
                pred_emd_delta_0, pred_emd_cls_0,
                pred_emd_delta_1, pred_emd_cls_1,
                bbox_targets, labels)
            loss1 = emd_loss_softmax(
                pred_emd_delta_1, pred_emd_cls_1,
                pred_emd_delta_0, pred_emd_cls_0,
                bbox_targets, labels)
            loss = torch.cat([loss0, loss1], axis=1)
            # requires_grad = False
            _, min_indices = loss.min(axis=1)
            loss_emd = loss[torch.arange(loss.shape[0]), min_indices]
            loss_emd = loss_emd.mean()
            loss_dict = {}
            loss_dict['loss_rcnn_emd'] = loss_emd*1.5  #10 ->3->2
            return loss_dict
        else:
            class_num = pred_emd_cls_0.shape[-1] - 1
            tag = torch.arange(class_num).type_as(pred_emd_cls_0) + 1
            tag = tag.repeat(pred_emd_cls_0.shape[0], 1).reshape(-1, 1)
            pred_scores_0 = F.softmax(pred_emd_cls_0, dim=-1)[:, 1:].reshape(-1, 1)
            pred_scores_1 = F.softmax(pred_emd_cls_1, dim=-1)[:, 1:].reshape(-1, 1)
            pred_delta_0 = pred_emd_delta_0[:, 4:].reshape(-1, 4)
            pred_delta_1 = pred_emd_delta_1[:, 4:].reshape(-1, 4)
            base_rois = rcnn_rois[:, 1:5].repeat(1, class_num).reshape(-1, 4)
            pred_bbox_0 = restore_bbox(base_rois, pred_delta_0, True)
            pred_bbox_1 = restore_bbox(base_rois, pred_delta_1, True)
            pred_bbox_0 = torch.cat([pred_bbox_0, pred_scores_0, tag], axis=1)
            pred_bbox_1 = torch.cat([pred_bbox_1, pred_scores_1, tag], axis=1)
            pred_bbox = torch.cat((pred_bbox_0, pred_bbox_1), axis=1)
            return pred_bbox


class Density(nn.Module):
    def __init__(self, input_channal):
        super().__init__()
        self.density = nn.Sequential(
            nn.Conv2d(input_channal, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 3, 1, 1),
            nn.ReLU(inplace=True),
        )

        self.criterion = nn.MSELoss(reduction='mean').cuda()

    def forward(self, featrue, all_num=0, target=None, mask=None):
        if target is None:
            density = self.density(featrue)
            return density

        self.criterion.to(featrue.device)

        density = self.density(featrue)
        if mask is not None:
            tem_per_num = torch.sum(density, dim=[1, 2, 3])
            loss_num = F.smooth_l1_loss(tem_per_num, all_num).to(featrue.device)
            loss = self.criterion(density * mask, target * mask)
        else:
            loss = self.criterion(density, target)
        loss_dict = {}
        loss_dict['loss_density'] = loss*0.1  #sssss-2 ssssss->
        loss_dict['loss_num'] = loss_num * 0.0001
        return loss_dict
def restore_bbox(rois, deltas, unnormalize=True):
    if unnormalize:
        std_opr = torch.tensor(config.bbox_normalize_stds[None, :]).type_as(deltas)
        mean_opr = torch.tensor(config.bbox_normalize_means[None, :]).type_as(deltas)
        deltas = deltas * std_opr
        deltas = deltas + mean_opr
    pred_bbox = bbox_transform_inv_opr(rois, deltas)
    return pred_bbox