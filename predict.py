import torch
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
from source.utils import load_model
from source.models.model_factory import Model
from main import Config
from source.dataset import SplitDataset, get_data
from mpl_toolkits.axes_grid1 import make_axes_locatable

def inspect_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    print("Checkpoint keys:", checkpoint.keys())
    return checkpoint

def load_model_on_cpu(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    if 'net_param' in checkpoint:
        model.load_state_dict(checkpoint['net_param'])
    else:
        model.load_state_dict(checkpoint)
    return model

def normalize_image(img):
    """Normalize the image to the range [0, 1] or [0, 255] depending on its type"""
    if img.min() < 0 or img.max() > 1:
        img = (img - img.min()) / (img.max() - img.min())
    return img

def show_image(dataset, configs, index=1):
    if not os.path.exists('result_images'):
        os.makedirs('result_images')
    model_names = ['input', 'unet_predrnn_v2_attention', 'unet_predrnn_attention', 'unet_predrnn_v2', 'unet_predrnn']

    for name in model_names:
        if name == 'input':
            fig, a = plt.subplots(2, 3, constrained_layout=True, figsize=(12, 8))
            sample, mask = dataset[index]
            sample = sample.unsqueeze(0)
            for i in range(configs.input_length):
                img = sample[0, i, :, :, :].numpy()
                img = normalize_image(img)
                im = a[0][i].imshow(img)
                a[0][i].axis('off')
                a[0][i].set_title('t - ' + str(configs.input_length - 1 - i), fontsize=24)

            for i in range(configs.pred_length):
                img = sample[0, i + configs.input_length, :, :, :].numpy()
                img = normalize_image(img)
                im = a[1][i].imshow(img)
                a[1][i].axis('off')
                a[1][i].set_title('t + ' + str(i + 1), fontsize=24)

            cb = fig.colorbar(im, ax=a.ravel().tolist(), shrink=0.75)
            cb.set_label(label='rain level', size=24)
            fig.savefig('result_images/groundtruth.png')
        else:
            fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
            configs.model_name = name
            model = Model(configs)  
            if name == 'unet_predrnn_v2_attention':
                model = load_model_on_cpu(model, f'checkpoints/{name}/model-160.ckpt')  
            if name == 'unet_predrnn_attention':
                model = load_model_on_cpu(model, f'checkpoints/{name}/model-190.ckpt')  
            if name == 'unet_predrnn_v2':
                model = load_model_on_cpu(model, f'checkpoints/{name}/model-170.ckpt') 
            if name == 'unet_predrnn':
                model = load_model_on_cpu(model, f'checkpoints/{name}/model-145.ckpt') 
            with torch.no_grad():
                sample, mask = dataset[index]
                sample = sample.unsqueeze(0).to(configs.device)
                mask = mask.unsqueeze(0).to(configs.device)
                pred = model(sample, mask)

                img = pred[0, 0, :, :, :].cpu().numpy() * 255
                img = normalize_image(img)
                im = ax.imshow(img)
                ax.axis('off')
                ax.set_title('t + 1', fontsize=24)

                cb = fig.colorbar(im, ax=ax, shrink=0.75)
                cb.set_label(label='rain level', size=24)
            fig.savefig('result_images/' + name + '.png')

if __name__ == '__main__':
    configs = Config()
    configs.total_length = 4
    configs.input_length = 3
    configs.batch_size = 1
    configs.pred_length = 1
    configs.device = torch.device('cpu')
    data = get_data(configs)
    dataset = SplitDataset(data, configs)
    show_image(dataset, configs, 60)
    print('Done!')
