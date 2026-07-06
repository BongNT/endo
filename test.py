import torch

# 1. First, define your model class (must be the same architecture as the saved one)

# 2. Load the state_dict
checkpoint = torch.load("/home/bongmedai/Endo/ultralytics/runs/segment/train5/weights/best.pt", weights_only=False)
print("Checkpoint keys:", checkpoint.keys())
print(checkpoint["epoch"])
print(checkpoint["train_args"])
# print(checkpoint["train_results"])
