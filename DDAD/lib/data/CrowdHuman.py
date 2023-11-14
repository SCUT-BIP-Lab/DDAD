import os
import cv2
import torch
import numpy as np
import scipy.io as io
from utils import misc_utils
from scipy.ndimage import filters
from sklearn.neighbors import NearestNeighbors
import h5py
class CrowdHuman(torch.utils.data.Dataset):
    def __init__(self, config, if_train):
        if if_train:
            self.training = True
            source = config.train_source    #train.odgt路径
            self.short_size = config.train_image_short_size   #800
            self.max_size = config.train_image_max_size  #1400
        else:
            self.training = False
            source = config.eval_source   #val_odgt路径
            self.short_size = config.eval_image_short_size
            self.max_size = config.eval_image_max_size
        self.records = misc_utils.load_json_lines(source)
        self.config = config

    def __getitem__(self, index):
        return self.load_record(self.records[index])

    def __len__(self):
        return len(self.records)

    def load_record(self, record):
        if self.training:
            if_flap = np.random.randint(2) == 1
        else:
            if_flap = False
        # image
        image_path = os.path.join(self.config.image_folder, record['ID']+'.jpg')
        image = misc_utils.load_img(image_path)
        image_h = image.shape[0]
        image_w = image.shape[1]
        self_creat_mask = torch.ones(image_h,image_w)
        if if_flap:
            image = cv2.flip(image, 1)
        mat_path_flap = []

        if self.training:
            # ground_truth
            gtboxes,img_mask = misc_utils.load_gt(record, 'gtboxes', 'fbox', self.config.class_names, self_creat_mask)
            keep = (gtboxes[:, 2]>=0) * (gtboxes[:, 3]>=0)   #有bbox的宽高数据，基本keep都是1
            gtboxes=gtboxes[keep, :]   #排除掉没有bbox的行
            gtboxes[:, 2:4] += gtboxes[:, :2]
            if if_flap:
                gtboxes = flip_boxes(gtboxes, image_w)
            mat_list = "/home/data/CrowdHuman/Midu_mat/Train_midu/" + record['ID'] + ".mat"  # mat
            mat_path_flap.append([mat_list,if_flap])
            nr_gtboxes = gtboxes.shape[0]   #有多少个gtboxes
            im_info = np.array([0, 0, 1, image_h, image_w, nr_gtboxes])
            return image, gtboxes, mat_path_flap, img_mask,im_info
        else:
            # image
            t_height, t_width, scale = target_size(
                    image_h, image_w, self.short_size, self.max_size)
            # INTER_CUBIC, INTER_LINEAR, INTER_NEAREST, INTER_AREA, INTER_LANCZOS4
            resized_image = cv2.resize(image, (t_width, t_height), interpolation=cv2.INTER_LINEAR)
            resized_image = resized_image.transpose(2, 0, 1)
            image = torch.tensor(resized_image).float()
            gtboxes,img_mask = misc_utils.load_gt(record, 'gtboxes', 'fbox', self.config.class_names,self_creat_mask)
            gtboxes[:, 2:4] += gtboxes[:, :2]
            gtboxes = torch.tensor(gtboxes)
            mat_list = "/home/data/CrowdHuman/Midu_mat/Val_midu/" + record['ID'] + ".mat"  # mat
            mat_path_flap.append([mat_list, if_flap])
            nr_gtboxes = gtboxes.shape[0]
            im_info = torch.tensor([t_height, t_width, scale, image_h, image_w, nr_gtboxes])
            return image, gtboxes, mat_path_flap,img_mask, im_info, record['ID']

    def create_density(self,L2s,d_map_h,d_map_w,image_size,scale=1, source_img_size= None):
        down_sample_ratio = 4
        map_shape = [d_map_h // down_sample_ratio, d_map_w // down_sample_ratio]  # 密度估计图尺寸
        res = np.zeros(shape=map_shape, dtype=np.float32)
        count_num = 0
        for k in range(len(L2s)):
            L2 = L2s[k]
            L2 = (int(L2[0]*scale), int(L2[1]*scale))
            if (L2[0] > image_size[1]) or (L2[1] > image_size[0]):
                continue
            if (L2[0] <0 ) or (L2[1] <0):
                continue
            res[min(map_shape[0] - 1, np.int16(L2[1] / down_sample_ratio))][
                min(map_shape[1] - 1, np.int16(L2[0] / down_sample_ratio))] += 1.0  # 人群位置图
        density = filters.gaussian_filter(res, 1)
        density = np.array(density)
        count_num = len(L2s)
        return density,count_num


    def merge_batch(self, data):
        images = [it[0] for it in data]
        image_size = [list(img.shape) for img in images]
        gt_boxes = [it[1] for it in data]   #xyxy
        mat_paths = [it[2] for it in data]
        img_masks = [it[3] for it in data]
        im_info = np.array([it[4] for it in data])
        batch_height = np.max(im_info[:, 3])  #最大的box高度
        batch_width = np.max(im_info[:, 4])  #最大的box宽度
        padded_images = [pad_image(         #用均值pad，图像在左上方，pad的均值在下右部分
                im, batch_height, batch_width, self.config.image_mean) for im in images]
        t_height, t_width, scale = target_size(
                batch_height, batch_width, self.short_size, self.max_size)  # 800 1400
        resized_images = np.array([cv2.resize(
                im, (t_width, t_height), interpolation=cv2.INTER_LINEAR) for im in padded_images])
        resized_images = resized_images.transpose(0, 3, 1, 2)
        images = torch.tensor(resized_images).float()
        all_denisty = []
        all_masks = []
        all_num = []
        for tem_num in range(len(mat_paths)):
            mat_path, flip_flag = mat_paths[tem_num][0]
            mat_data = io.loadmat(mat_path)
            gt_mat = mat_data["image_info"][0][0][0][0]
            if len(gt_mat[0]) == 1:
                pts = gt_mat[1]
            else:
                pts = gt_mat[0]
            if flip_flag:
                pts = flip_piont(pts, image_size[tem_num][1])

            density,person_num = self.create_density(pts,t_height,t_width,(t_height,t_width),scale)
            tem_mask = img_masks[tem_num].clone().float().unsqueeze(0).unsqueeze(0)
            img_mask = torch.nn.functional.interpolate(tem_mask, size=(t_height//4, t_width//4), scale_factor=None,
                                                       mode='nearest', align_corners=None)
            img_mask = img_mask.squeeze(1)
            if flip_flag:
                img_mask = torch.flip(img_mask, dims=(2,))
            all_masks.append(img_mask)
            all_num.append(person_num)

            all_denisty.append(torch.tensor(density.copy()).float())
        all_num = torch.tensor(all_num).float()  # 变换之后的GT

        all_masks = torch.stack(all_masks, dim=0)
        all_denisty = torch.stack(all_denisty, dim=0).unsqueeze(1)
        # ground_truth
        ground_truth = []
        for it in gt_boxes:
            gt_padded = np.zeros((self.config.max_boxes_of_image, self.config.nr_box_dim))   #最多每张图片500个人，维度为5
            it[:, 0:4] *= scale
            max_box = min(self.config.max_boxes_of_image, len(it))
            gt_padded[:max_box] = it[:max_box]
            ground_truth.append(gt_padded)
        ground_truth = torch.tensor(np.array(ground_truth)).float()   #变换之后的GT
        debug_gt =False
        if debug_gt==True:
            out_target = all_denisty.cpu().data.numpy()
            image = images.permute(0,2,3,1).cpu().data.numpy()
            for target_show,image_show in zip(out_target, image):
                image_show = image_show
            print("crowdhuman-=-=-=-=--len_gt=:",len(ground_truth[0]))
            print("crowdhuman-=-=-=-=--im_info=:",im_info)
            print("crowdhuman-=-=-=-=--image=:",image.shape)
            print("crowdhuman-=-=-=-=--image_show=:",image_show.shape)
            print("crowdhuman-=-=-=-=--mat_path=:",mat_path)
            imgname = './save_npys/'+mat_path.split('/')[-1].split('.')[0]+'.png'
            cv2.imwrite(imgname, image_show)
            for twx in range(len(ground_truth[0])):
                tem_loc =  ground_truth[0][twx]
                if tem_loc[4] ==1:
                    print("crowdhuman-=-=-=-=--ground_truth=:", ground_truth[0][twx])
                    print("yes")
                    cv2.rectangle(image_show, (tem_loc[0], tem_loc[1]), (tem_loc[2], tem_loc[3]), (0, 0, 255), 2)
            imgnamegt = './save_npys/' + mat_path.split('/')[-1].split('.')[0] + 'gt.png'
            cv2.imwrite(imgnamegt, image_show)

            out_density = all_denisty.cpu().data.numpy()
            out_target = out_density
            H, W = out_target.shape[-2:]

            for density_show,  image_show in zip(out_target, image):
                image_show = cv2.resize(image_show, (W, H))
                density_map = density_show[0]
                X = np.uint8(255 * (density_map - np.min(density_map)) / (np.max(density_map) - np.min(density_map)))
                G1 = cv2.applyColorMap(X, 2)
                imgnamegt = './save_npys/' + mat_path.split('/')[-1].split('.')[0] + 'density_img.png'
                img_show = np.concatenate([G1, image_show], axis=1)
                cv2.imwrite(imgnamegt, img_show)
        # im_info
        im_info[:, 0] = t_height
        im_info[:, 1] = t_width
        im_info[:, 2] = scale
        im_info = torch.tensor(im_info)
        if max(im_info[:, -1] < 2):
            return None, None, None, None,None,None
        else:
            return images, ground_truth, all_denisty,all_masks,all_num, im_info

def target_size(height, width, short_size, max_size):
    im_size_min = np.min([height, width])
    im_size_max = np.max([height, width])
    scale = (short_size + 0.0) / im_size_min
    if scale * im_size_max > max_size:
        scale = (max_size + 0.0) / im_size_max
    t_height, t_width = int(round(height * scale)), int(
        round(width * scale))
    return t_height, t_width, scale

def flip_boxes(boxes, im_w):
    flip_boxes = boxes.copy()
    for i in range(flip_boxes.shape[0]):
        flip_boxes[i, 0] = im_w - boxes[i, 2] - 1
        flip_boxes[i, 2] = im_w - boxes[i, 0] - 1
    return flip_boxes

def flip_piont(point, im_w):
    point = point.copy()
    for i in range(point.shape[0]):
        point[i, 0] = im_w - point[i, 0] - 1
    return point

def pad_image(img, height, width, mean_value):
    o_h, o_w, _ = img.shape
    margins = np.zeros(2, np.int32)
    assert o_h <= height
    margins[0] = height - o_h
    img = cv2.copyMakeBorder(
        img, 0, margins[0], 0, 0, cv2.BORDER_CONSTANT, value=0)
    img[o_h:, :, :] = mean_value
    assert o_w <= width
    margins[1] = width - o_w
    img = cv2.copyMakeBorder(
        img, 0, 0, 0, margins[1], cv2.BORDER_CONSTANT, value=0)
    img[:, o_w:, :] = mean_value
    return img
