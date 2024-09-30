import torch

import torch
for i in range(torch.cuda.device_count()):
   print(torch.cuda.get_device_properties(i).name)