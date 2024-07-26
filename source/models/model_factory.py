import torch
import torch.nn.functional as F
from source.models import unet_predrnn_v2_attention, unet_predrnn, unet_predrnn_v2, unet_predrnn_attention


class Model(torch.nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        networks_map = {
            'unet_predrnn': unet_predrnn.RNN,
            'unet_predrnn_v2': unet_predrnn_v2.RNN,
            'unet_predrnn_attention': unet_predrnn_attention.RNN,
            'unet_predrnn_v2_attention': unet_predrnn_v2_attention.RNN,
        }
        self.num_hidden = configs.num_hidden
        self.num_layers = len(self.num_hidden)
        if configs.model_name in networks_map:
            Network = networks_map[configs.model_name]
            self.net = Network(self.num_layers, self.num_hidden, configs).to(configs.device)
        else:
            raise ValueError('Name of network unknown %s' % configs.model_name)

    def forward(self, x, mask):
        """
        :param x: (b, 5, img_size, img_size, 1)
        :param mask: (b, 4, img_size, img_size, 1)
        :return: (b, 19, img_size, img_size, 1)
        """
        return self.net(x, mask)

    def training_step(self, batch, batch_ix):
        x, mask = batch
        pred = self.forward(x, mask)
        loss = F.mse_loss(pred, x[:, 1:])
        self.log('train_loss', loss)
        return loss
