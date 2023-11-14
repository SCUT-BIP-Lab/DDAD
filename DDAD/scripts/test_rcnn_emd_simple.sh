cd ..
cd tools

CUDA_VISIBLE_DEVICES=0,1 python train.py -md rcnn_emd_simple


CUDA_VISIBLE_DEVICES=0 python test.py -md rcnn_emd_simple -r 68





