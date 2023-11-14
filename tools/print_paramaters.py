
import torch
pretrained_state = torch.load('/home/.../paper_code/CrowdDet_upmid_denisty/model/rcnn_emd_simple/outputs/model_dump/dump-75.pth')
model_paras = pretrained_state['state_dict']


for key in model_paras:
    print(key)
    if key.split('.')[0]=="Density" or key.split('.')[0]=="RCNN":
        print(key)

        print(model_paras[key])
        if key.split('.')[-1]=="weight":
            with open("right_GT.txt", "a") as f:
                f.write(str(key))
                f.write(str(model_paras[key]))
    else:
        continue