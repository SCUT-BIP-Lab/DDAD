import torch

# 加载整个模型
model = torch.load('/home/lz/workspace/DDAD/model/rcnn_emd_simple/outputs/model_dump/dump-67.pth', map_location=torch.device('cpu'))

# 提取模型字典中的权重
model_weights = model['state_dict'] if 'state_dict' in model else model

# 创建新的字典，只保留 'state_dict' 键和对应的权重
new_model_dict = {'state_dict': model_weights}

# 保存只包含 'state_dict' 的模型
torch.save(new_model_dict, '/home/lz/workspace/DDAD/model/rcnn_emd_simple/outputs/model_dump/dump-68.pth')
