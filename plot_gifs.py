from src.SegNet import SegNet
from src.Dataset_Seg import get_datasets_seg
from src.utils import SQM, gifify, get_all_SQM
from train_net import valid_fn
import os

# Parameters
epoch_to_load = 'None'  # 'None' if take last checkpoint
model_name = 'PredNet_TA1_JP1-0_PR1-0_SM3-0_SB1-0_SD1-0_CB1-0_CD1-0_AC(3-16-32)_RC(16-32-64)_RL(h-h-h)_FS(3-3-3)_PL(1-1-1)_SL(0-1-0)_CL(0-0-1)'
h5_path = 'data/training_room_dataset_03.h5' # dataset path
n_samples = 1000  # number of samples in the dataset
tr_ratio = 0.85 # only validation samples
n_samples_to_plot = 10
speed_up_factor = 1
remove_ground = True

def main():
    print(f'Loading model: {model_name}')
    model, _, _, _, _ = SegNet.load_model(model_name, epoch_to_load)
    print('Loading a few samples...')
    _, test_dataloader = get_datasets_seg(
        dataset_path=h5_path, tr_ratio=tr_ratio, n_samples=n_samples,
        batch_size_train=0, batch_size_valid=n_samples_to_plot, mode='test')
    print('Generating gifs...')
    valid_fn(test_dataloader, model, plot_gif=True)
    print('Done!')

if __name__ == '__main__':
    main()
