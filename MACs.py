import torch
from main import Config
from source.models import predrnn_v2, predrnn, predrnn_v3, attention_predrnn, unet_predrnn, unet_predrnn_v2
from thop import profile

# Create a network and a corresponding input
configs = Config()
device = 'cuda:0'
model = unet_predrnn_v2.RNN(len(configs.num_hidden), configs.num_hidden, configs).to(device)
inp = torch.rand(2, 10, 128, 128, 1).to(device)
mask = torch.rand(2, 4, 128, 128, 1).to(device)

# Count the number of FLOPs
macs, params = profile(model, inputs=(inp, mask))
from thop import clever_format
macs, params = clever_format([macs, params], "%.3f")
print(macs, params)
