import os
import torch
import numpy as np
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch.utils.data import Dataset

# Import your other modules
from source.utils import load_model
from source.models.model_factory import Model
from main import Config

# Dataset and DataLoader functions
class SplitDataset(Dataset):
    def __init__(self, data, configs):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.configs = configs

    def __getitem__(self, item):
        inputs = self.data[item: item + self.configs.total_length]
        mask_true = torch.zeros((self.configs.pred_length - 1, self.configs.img_width, self.configs.img_width, 3),
                                dtype=torch.float32)
        return inputs, mask_true

    def __len__(self):
        return len(self.data) - self.configs.total_length

# Utility functions
def load_model_on_cpu(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    if 'net_param' in checkpoint:
        model.load_state_dict(checkpoint['net_param'])
    else:
        model.load_state_dict(checkpoint)
    return model

def normalize_image(img):
    if img.min() < 0 or img.max() > 1:
        img = (img - img.min()) / (img.max() - img.min())
    return img

def preprocess_images(images, configs):
    sample_data = []
    for image in images:
        image = image.convert('RGB')
        image = image.resize((configs.img_width, configs.img_width))
        data = np.array(image).astype(np.float32)
        sample_data.append(data)
    sample_data = np.stack(sample_data, 0)
    return sample_data

def show_image(sample, mask, model, configs, name):
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    with torch.no_grad():
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
    return fig

# Streamlit app
def main():
    st.title("Weather Prediction Demo")

    configs = Config()
    configs.total_length = 4
    configs.input_length = 3
    configs.batch_size = 1
    configs.pred_length = 1
    configs.device = torch.device('cpu')
    configs.img_width = 128  # You might want to adjust this as necessary
    configs.channel = 3  # Setting input channels to 3 for RGB

    uploaded_files = st.file_uploader("Choose 3 images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    if len(uploaded_files) != 3:
        st.warning("Please upload exactly 3 images.")
        return

    model_names = ['unet_predrnn_attention', 'unet_predrnn_v2_attention', 'unet_predrnn_v2', 'unet_predrnn']
    selected_model = st.selectbox('Select a model', model_names)

    if st.button('Predict'):
        images = [Image.open(file) for file in uploaded_files]
        data = preprocess_images(images, configs)
        dataset = SplitDataset(data, configs)

        model = Model(configs)
        if selected_model == 'unet_predrnn_attention':
            model = load_model_on_cpu(model, f'checkpoints/{selected_model}/model-190.ckpt')
        if selected_model == 'unet_predrnn_v2_attention':
            model = load_model_on_cpu(model, f'checkpoints/{selected_model}/model-160.ckpt')
        if selected_model == 'unet_predrnn_v2':
            model = load_model_on_cpu(model, f'checkpoints/{selected_model}/model-170.ckpt')
        if selected_model == 'unet_predrnn':
            model = load_model_on_cpu(model, f'checkpoints/{selected_model}/model-145.ckpt')
        sample, mask = dataset[0]
        fig = show_image(sample, mask, model, configs, selected_model)
        st.pyplot(fig)

if __name__ == '__main__':
    main()