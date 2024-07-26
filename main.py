import torch
import numpy as np
import time
import os
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from source.dataset import get_data, SplitDataset
from source.models.model_factory import Model
from tensorboardX import SummaryWriter
from source.utils import save_model

class Config():
    model_name = 'unet_predrnn_v2_attention'
    path_root = '/Users/houngnt/Downloads/radar_data-main/Data/dataset/20200601'
    # for model definition
    channel = 3
    stride = 1
    num_hidden = [64]
    layer_norm = True
    img_width = 128
    filter_size = 3
    total_length = 4  # 3 input frames + 1 predicted frame
    input_length = 3  # Number of input frames
    pred_length = 1   # Number of frames to predict (4th frame)
    # for training stage
    split_data = 0.8
    batch_size = 16
    epochs = 200
    lr = 0.001
    eval_interval = 1
    save_interval = 5
    device = 'cpu'  # Set device to 'cpu'
    # for saving model
    save_dir = 'checkpoints'

if __name__ == '__main__':
    configs = Config()
    # for saving checkpoint and log
    if not os.path.exists('logs'):
        os.makedirs('logs')

    if not os.path.exists(configs.save_dir):
        os.makedirs(configs.save_dir)
    if not os.path.exists(os.path.join(configs.save_dir, configs.model_name)):
        os.makedirs(os.path.join(configs.save_dir, configs.model_name))

    summary = SummaryWriter('logs/' + configs.model_name + '_' + str(configs.batch_size) \
                            + '_' + str(configs.epochs) + '_' + str(configs.lr)
                            + '_' + str(configs.num_hidden))

    # get dataset
    data = get_data(configs)
    dataset = SplitDataset(data, configs)

    # train and test split
    tol_nums = len(dataset)
    train_nums = int(tol_nums * configs.split_data)
    trainds, testds = random_split(dataset, [train_nums, tol_nums - train_nums])
    traindl = DataLoader(trainds, batch_size=configs.batch_size, shuffle=True)
    testdl = DataLoader(testds, batch_size=configs.batch_size)

    # model definition
    model = Model(configs)
    model = model.to(configs.device)  # Ensure model is moved to the correct device
    # optimizer definition
    optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

    start = time.time()

    for epoch in range(configs.epochs):
        model.train()
        batch_losses = []
        for batch_ix, batch in tqdm(enumerate(traindl)):
            optimizer.zero_grad()
            x, mask = batch
            x = x.to(configs.device)  # Ensure data is moved to the correct device
            mask = mask.to(configs.device)
            pred = model.forward(x, mask)
            # For prediction length 1, compare with the actual next frame
            loss = F.mse_loss(pred, x[:, -1:])
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
            if (batch_ix + 1) % 20 == 0:
                print(f'epoch{epoch} batch ix:{batch_ix} train loss:{np.sum(batch_losses):.4f}')
                batch_losses = []
        summary.add_scalar('Train_loss', np.mean(batch_losses), epoch)
        # lr_scheduler.step()

        if epoch % configs.eval_interval == 0:
            with torch.no_grad():
                model.eval()
                test_losses = []
                for batch in testdl:
                    x, mask = batch
                    x = x.to(configs.device)  # Ensure data is moved to the correct device
                    mask = mask.to(configs.device)
                    pred = model.forward(x, mask)
                    loss = F.mse_loss(pred, x[:, -1:])
                    test_losses.append(loss.item())
                print(f'epoch{epoch} test loss:{np.mean(test_losses):.4f}')
            summary.add_scalar('Evaluation_loss', np.mean(test_losses), epoch)

        if epoch % configs.save_interval == 0:
            print('Saving checkpoint of model: ')
            save_model(model, epoch, configs)

    end = time.time()
    print("training time: ", end-start)
