import torch
import os


def save_model(network, epoch, configs):
    stats = {}
    stats['net_param'] = network.state_dict()
    checkpoint_path = os.path.join(configs.save_dir, configs.model_name, 'model' + '-' + str(epoch) + '.ckpt')
    torch.save(stats, checkpoint_path)
    print("save model to %s" % checkpoint_path)


def load_model(network, checkpoint_path):
    print('load model:', checkpoint_path)
    stats = torch.load(checkpoint_path, map_location='cpu')
    network.load_state_dict(stats['net_param'])
    return network
