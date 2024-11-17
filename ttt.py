#查看预训练权重文件的结构
import torch

# Define the path to the pre-trained weights file
weights_path = "/home/yuwenjing/DeepLearning/MNv2/supervised_suprem_swinunetr_2100.pth"

# Try loading only the keys of the weights file to inspect its structure
try:
    checkpoint = torch.load(weights_path, map_location='cpu')
    
    # Check if the checkpoint has a nested 'state_dict' or if it's a flat dictionary
    if 'net' in checkpoint:
        weights_dict = checkpoint['net']
    else:
        weights_dict = checkpoint
    
    # Print only the keys to avoid loading full tensor data into memory
    print("Keys in weights_dict:", list(weights_dict.keys()))

except Exception as e:
    print("Error loading weights:", e)
