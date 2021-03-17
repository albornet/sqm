import os
import torch
import numpy as np
import imageio
from src.SegNet import SegNet
from src.Dataset_Seg import get_datasets_seg

# Parameters
epoch_to_load = 'None'  # 'None' if take last checkpoint
model_name = 'THRESH_1/DROPOUT_01_01/PredNet_TA1_DM0_JP0-0_PR1-0_SM0-0_SB0-0_SD0-0_AC(3-16)_RC(16-64)_RL(h-h)_FS(5-5)_PL(1-1)_SL(0-0)'
h5_path = 'data/training_room_dataset_04.h5' # dataset path
n_samples = 1000  # number of samples in the dataset
tr_ratio = 0.85 # only validation samples
n_samples_to_plot = 100
speed_up_factor = 1
remove_ground = True

def plot_recons(I_seq, S_lbl, P_seq, S_seq, epoch=0, sample_indexes=[0]):
    img_plot = I_seq.detach().cpu().numpy()
    rec_plot = P_seq.detach().cpu().numpy()
    lbl_data = S_lbl.detach().cpu().numpy()  # [:, 1:, ...]
    seg_data = S_seq.detach().cpu().numpy()  # [:, 1:, ...]
    lbl_plot = np.zeros(((lbl_data.shape[0], 3) + lbl_data.shape[2:]))
    seg_plot = np.zeros(((seg_data.shape[0], 3) + seg_data.shape[2:]))
    for shift in [0, 1, 2]:
        lbl_plot[:, shift, ...] = lbl_data[:, shift::3, ...].sum(axis=1)
        seg_plot[:, shift, ...] = seg_data[:, shift::3, ...].sum(axis=1)
    batch_size, n_channels, n_rows, n_cols, n_frames = img_plot.shape
    h_rectangle = np.zeros((batch_size, n_channels, 10, n_cols, n_frames))
    v_rectangle = np.zeros((batch_size, n_channels, n_rows*2 + 10, 10, n_frames))
    dat_rec = np.clip(np.concatenate((img_plot, h_rectangle, rec_plot), axis=2), 0.0, 1.0)
    lbl_seg = np.clip(np.concatenate((lbl_plot, h_rectangle, seg_plot), axis=2), -10.0, 10.0)
    output_batch = np.concatenate((dat_rec, v_rectangle, lbl_seg), axis=3).transpose((0, 2, 3, 1, 4))
    for s in sample_indexes:
        output_seq = output_batch[s]
        gif_frames = [(255.*output_seq[..., t]).astype(np.uint8) for t in range(n_frames)]
        gif_path = f'./ckpt/{model_name}/testmode_epoch{epoch:02}_sample{s:02}'
        imageio.mimsave(f'{gif_path}.gif', gif_frames, duration=0.1)

def valid_fn(valid_dataloader, model):
    model.eval()
    plot_loss_valid = 0.0
    n_batches = len(valid_dataloader)
    with torch.no_grad():
        for i, (batch, S_lbl, _) in enumerate(valid_dataloader):
            if speed_up_factor > 1:
                zero_frame = np.random.randint(speed_up_factor)
                batch = batch[..., zero_frame::speed_up_factor]
                S_lbl = S_lbl[..., zero_frame::speed_up_factor]
            if remove_ground:
                S_lbl = S_lbl[:, 1:, ...]
            E_seq, R_seq, P_seq, S_seq = model(batch)
            plot_recons(batch, S_lbl, P_seq, S_seq, sample_indexes=range(len(batch)))
            exit()

def main():
    print(f'Loading model: {model_name}')
    model, _, _, _, _ = SegNet.load_model(model_name, epoch_to_load)
    print('Loading a few samples...')
    _, test_dataloader = get_datasets_seg(
        dataset_path=h5_path, tr_ratio=tr_ratio, n_samples=n_samples,
        batch_size_train=0, batch_size_valid=n_samples_to_plot, mode='test')
    print('Generating gifs...')
    valid_fn(test_dataloader, model)
    print('Done!')

if __name__ == '__main__':
    main()
