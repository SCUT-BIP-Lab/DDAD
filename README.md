# DDAD: Detachable Crowd Density Estimation Assisted Pedestrian Detection

This is the pytorch implementation of our paper [DDAD: Detachable Crowd Density Estimation Assisted Pedestrian Detection](https://ieeexplore.ieee.org/document/9963778), published in IEEE Transactions on Intelligent Transportation Systems (T-ITS) 2023.

# Quick start

## Installation
- Install dependencies:
```
pip install -r requirements.txt
```

- Training.
More training seetings can be set in ```config.py```
```
cd tools
python train.py -md rcnn_fpn_baseline
python train.py -md rcnn_emd_simple_idad
```

- Testing.
```
cd tools
python test.py -md rcnn_fpn_baseline -r 1
python test.py -md rcnn_emd_simple_idad -r 1
```
When testing different models, simply modify the numbers in the test commands. The result json file will be evaluated automatically.


Models are avaliable in the [model zoo](https://pan.baidu.com/s/1TRq3h9UYYSTW7NMNwnt93Q?pwd=4i2x), code: 4i2x.

Mat files are in the [mat files](https://pan.baidu.com/s/1OqyFcV6pGcueEw5IDRlvdw?pwd=fpga), code: fpga.





---
## BibTex

This is the repository for the paper MarginMatch: DDAD: Detachable Crowd Density Estimation Assisted Pedestrian Detection. If you found this repository helpful, consider citing our paper:

```
@article{tang2022ddad,
  title={DDAD: Detachable Crowd Density Estimation Assisted Pedestrian Detection},
  author={Tang, Wenxiao and Liu, Kun and Shakeel, M Saad and Wang, Hao and Kang, Wenxiong},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  volume={24},
  number={2},
  pages={1867--1878},
  year={2022},
  publisher={IEEE}
}
```


# Acknowledgement
- This code was inspired from [CrowdDet](https://github.com/xg-chu/CrowdDet).

