import torch
import torch.nn as nn
from source.layers.SpatioTemporalLSTMCell import SpatioTemporalLSTMCell
from source.layers.UnetBase import *

class Attention(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super(Attention, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(input_dim, attention_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(attention_dim, input_dim, kernel_size=1)  
        )

    def forward(self, x):
        attn_weights = self.attention(x)
        attn_weights = torch.sigmoid(attn_weights)
        return x * attn_weights

class RNN(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):
        super(RNN, self).__init__()

        self.configs = configs
        self.frame_channel = configs.channel
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []
        self.down1 = DoubleConv(self.frame_channel, self.num_hidden[0] // 4)
        self.down2 = Down(self.num_hidden[0] // 4, self.num_hidden[0] // 2)
        self.down3 = Down(self.num_hidden[0] // 2, self.num_hidden[0])

        bilinear = True
        factor = 2 if bilinear else 1
        width = self.num_hidden[0] // 2

        for i in range(num_layers):
            in_channel = self.num_hidden[0] if i == 0 else num_hidden[i - 1] * 2
            cell_list.append(
                SpatioTemporalLSTMCell(in_channel, num_hidden[i] * 2, width, configs.filter_size,
                                       configs.stride, configs.layer_norm)
            )
        self.cell_list = nn.ModuleList(cell_list)

        self.up1 = Up(num_hidden[num_layers - 1] * 2, self.num_hidden[0], bilinear)
        self.up2 = Up(num_hidden[num_layers - 1], self.num_hidden[0] // 2, bilinear)

        self.conv_last = nn.Conv2d(self.num_hidden[0] // 2, 1, kernel_size=1, stride=1, padding=0, bias=False)

        # Attention mechanism
        self.attention = Attention(1, self.num_hidden[0] // 4)

    def forward(self, frames, mask_true):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames = frames.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        batch = frames.shape[0]
        height = self.num_hidden[0] // 2
        width = self.num_hidden[0] // 2

        next_frames = []
        h_t = []
        c_t = []

        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i] * 2, height, width]).to(self.configs.device)
            h_t.append(zeros)
            c_t.append(zeros)

        memory = torch.zeros([batch, self.num_hidden[0] * 2, height, width]).to(self.configs.device)

        for t in range(self.configs.total_length - 1):
            if t < self.configs.input_length:
                net = frames[:, t]
            else:
                net = mask_true[:, t - self.configs.input_length] * frames[:, t] + \
                      (1 - mask_true[:, t - self.configs.input_length]) * x_gen3
            net1 = self.down1(net)
            net2 = self.down2(net1)  # 64x64x64
            net3 = self.down3(net2)  # 64x64x64
            h_t[0], c_t[0], memory = self.cell_list[0](net3, h_t[0], c_t[0], memory)
            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)
            x_gen1 = self.up1(h_t[self.num_layers - 1])
            x_gen2 = self.up2(x_gen1)
            x_gen3 = self.conv_last(x_gen2)
            # Apply attention mechanism
            x_gen3 = self.attention(x_gen3)
            next_frames.append(x_gen3)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        return next_frames
