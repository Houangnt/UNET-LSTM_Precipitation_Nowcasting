import torch
import torch.nn as nn
import torch.nn.functional as F
from source.layers.SpatioTemporalLSTMCell_v2 import SpatioTemporalLSTMCell
from source.layers.UnetBase import *


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

        # width = configs.img_width


        for i in range(num_layers):
            in_channel = self.num_hidden[0] if i == 0 else num_hidden[i - 1] * 2
            cell_list.append(
                SpatioTemporalLSTMCell(in_channel, num_hidden[i]*2, width, configs.filter_size,
                                       configs.stride, configs.layer_norm)
            )
        self.cell_list = nn.ModuleList(cell_list)

        self.up1 = Up(num_hidden[num_layers - 1] * 2, self.num_hidden[0], bilinear)
        self.up2 = Up(num_hidden[num_layers - 1], self.num_hidden[0] // 2, bilinear)

        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1] // 2, self.frame_channel, kernel_size=1, stride=1, padding=0,
                                   bias=False)
        # shared adapter
        adapter_num_hidden = num_hidden[0]
        self.adapter = nn.Conv2d(adapter_num_hidden*2, adapter_num_hidden*2, 1, stride=1, padding=0, bias=False)

    def forward(self, frames_tensor, mask_true):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        batch = frames.shape[0]
        height = self.num_hidden[0] // 2
        width = self.num_hidden[0] // 2
        next_frames = []
        h_t = []
        c_t = []
        delta_c_list = []
        delta_m_list = []

        decouple_loss = []

        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i]*2, height, width]).to(self.configs.device)
            h_t.append(zeros)
            c_t.append(zeros)
            delta_c_list.append(zeros)
            delta_m_list.append(zeros)

        memory = torch.zeros([batch, self.num_hidden[0]*2, height, width]).to(self.configs.device)

        for t in range(self.configs.total_length - 1):
            if t < self.configs.input_length:
                net = frames[:, t]
            else:
                net = mask_true[:, t - self.configs.input_length] * frames[:, t] + \
                      (1 - mask_true[:, t - self.configs.input_length]) * x_gen3
            net1 = self.down1(net)
            net2 = self.down2(net1)  # 64x64x64
            net3 = self.down3(net2)  # 64x64x64
            h_t[0], c_t[0], memory, delta_c, delta_m = self.cell_list[0](net3, h_t[0], c_t[0], memory)
            delta_c_list[0] = F.normalize(self.adapter(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1), dim=2)
            delta_m_list[0] = F.normalize(self.adapter(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1), dim=2)

            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory, delta_c, delta_m = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)
                delta_c_list[i] = F.normalize(self.adapter(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1), dim=2)
                delta_m_list[i] = F.normalize(self.adapter(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1), dim=2)
            x_gen1 = self.up1(h_t[self.num_layers - 1])
            x_gen2 = self.up2(x_gen1)
            x_gen3 = self.conv_last(x_gen2)
            next_frames.append(x_gen3)
            # decoupling loss
            for i in range(0, self.num_layers):
                decouple_loss.append(
                    torch.mean(torch.abs(torch.cosine_similarity(delta_c_list[i], delta_m_list[i], dim=2))))

        decouple_loss = torch.mean(torch.stack(decouple_loss, dim=0))
        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]

        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        # loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:]) + self.configs.decouple_beta * decouple_loss
        return next_frames
