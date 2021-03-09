import torch
import numpy as np
import time
import imageio
import matplotlib.pyplot as plt
from src.SegNet import SegNet
from src.Dataset_Seg import get_datasets_seg
from src.Jacobian_penalty import jacobian_penalty

# Training parameters
device = 'cuda'            # 'cuda' or 'cpu' ('cpu' never tested)
load_model = False         # if False, create a new networks
epoch_to_load = None       # None for last epoch; not used if load_model == False
n_epochs_to_run = 20       # from the last epoch if load_model == True
n_epoch_save = 5           # will save a new checkpoint every n_epoch_save
learning_rate = 0.001      # is modified by the scheduler
learning_rate_schedule = {'milestones': list(range(0, 2*n_epochs_to_run, 5)), 'gamma': 0.5}
P_red_loss_weight = 1.0    # unsupervised loss (try running and adjust)
S_mse_loss_weight = 3.0    # supervised loss (segmentation, mean square error)
S_bce_loss_weight = 1.0    # supervised loss (segmentation, cross-entropy)
S_dic_loss_weight = 1.0    # supervised loss (segmentation, 1.0 - dice score)
J_pen_loss_weight = 0.0    # unsupervised loss (convergence to dynamical stability)
batch_size_train = 16      # try larger and larger values, until it does not fit
batch_size_valid = 96      # faster because no need to loss.backward()
bce_loss_fn = torch.nn.BCELoss()

# Network parameters
R_channels = ( 8, 16, 32, 64, 128) #, 256)                  # for each layer, top-down feature dimensionality
A_channels = ( 3,  8, 16, 32,  64) #, 128)                  # for each layer, bottom-up feature dimensionality
R_layers = ('hgru', 'hgru', 'hgru', 'hgru', 'hgru') #, 'hgru')  # for each layer, type of representation
P_layers = (1, 1, 1, 1, 1) #, 1)                           # for each layer, which E are used in the pred_loss
S_layers = (1, 1, 1, 1, 1) #, 1)                           # for each layer, whether R output is sent to the segmentation layer
J_layers = (0, 0, 0, 0, 0) #, 0)                           # which layer is used in jacobian penalty calculations
J_times = []                                         # which time_steps are used for Jacobian penalty calculations
filter_sizes = (5, 5, 5, 5, 5) #, 5)                       # between each layer, size of the convolutions
do_time_aligned = True                               # if True, take neuronal delays into account
do_dopamine_mode = False                             # pred loss triggers learning in other losses (factor of lr)
do_prediction = P_red_loss_weight > 0.0
do_segmentation = (S_mse_loss_weight + S_bce_loss_weight + S_dic_loss_weight) > 0.0
do_jacobian_penalty = J_pen_loss_weight > 0
if do_jacobian_penalty:
  mu = 0.5  # jacobain penalty scaling parameter
if do_dopamine_mode:
    do_prediction = True  # even if pred weight is 0.0
model_name = f'PredNet_TA{int(do_time_aligned)}_DM{int(do_dopamine_mode)}_JP{J_pen_loss_weight}'\
            +f'_PR{P_red_loss_weight}_SM{S_mse_loss_weight}_SB{S_bce_loss_weight}'\
            +f'_SD{S_dic_loss_weight}_AC{A_channels}_RC{R_channels}_RL{tuple([r[0] for r in R_layers])}'\
            +f'_FS{filter_sizes}_PL{P_layers}_SL{S_layers}'
model_name = model_name.replace('.', '-').replace(',', '-').replace(' ', '').replace("'", '')
P_layers = [l for l, e in enumerate(P_layers) if e == 1]  # for convenience, e.g. [1, 0, 0, 1] --> [0, 3]
S_layers = [l for l, e in enumerate(S_layers) if e == 1]  # for convenience, e.g. [1, 0, 0, 1] --> [0, 3]
J_layers = [l for l, e in enumerate(J_layers) if e == 1]  # for convenience, e.g. [1, 0, 0, 1] --> [0, 3]

# Dataset parameters
h5_path = 'data/training_room_dataset_04.h5'  # dataset path ()
n_samples = 10000                             # number of samples in the dataset
n_classes = 4                                 # n_object_types + 1 (for "nothing")
tr_ratio = 0.85                               # training vs validation ratio 0.85/0.15
speed_up_factor = 1                           # every speed_up frame given to the network (random first frame)
t_start_loss = 1                              # number of frames ignored in the loss, after first one
occlusion = 0                                 # add occluding bars to all images
augmentation = 1                              # data augmentation on the dataset (albumentations)  slowdown???
dvs_mode = 0                                  # use dvs image samples (else normal images) POS IS [255 255 0], NEG IS [0 255 0]
remove_ground = 1                             # have no class for groung in segmentation labels
if remove_ground:
    n_classes -= 1


def plot_recons(I_seq, S_lbl, P_seq, S_seq, epoch=0, sample_indexes=[0], mode='train'):
    
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
        gif_path = f'./ckpt/{model_name}/{mode}mode_epoch{epoch:02}_sample{s:02}'
        imageio.mimsave(f'{gif_path}.gif', gif_frames, duration=0.1)

### TEST ###
def loss_fn(E_seq, R_seq, S_seq, S_lbl, batch_idx, n_batches, mode='train'):

    # Initialize losses
    batch_size, n_frames = E_seq[0][0].shape[0], len(E_seq[0])
    P_red_loss = torch.zeros((batch_size,)).cuda()
    S_dic_loss = torch.zeros((batch_size,)).cuda()
    S_bce_loss = torch.zeros((batch_size,)).cuda()
    S_mse_loss = torch.zeros((batch_size,)).cuda()
    J_pen_loss = torch.zeros((batch_size,)).cuda()

    # First, unsupervised losses
    J_pen_signal = torch.zeros((batch_size, n_frames))
    D_opa_signal = 1.0/(n_frames-t_start_loss)*torch.ones((batch_size, n_frames)).cuda()
    for t in range(t_start_loss, n_frames):       
    
        # Prediction loss E_seq is [n_frames, n_layers][batch_size, a_channels[l]*2, w, h]
        if do_prediction or do_dopamine_mode:
            P_red_loss_t = torch.zeros((batch_size,)).cuda()
            for l in P_layers:
                P_red_loss_t += E_seq[l][t].mean(axis=(-3, -2, -1))
            P_red_loss_t = P_red_loss_t/len(P_layers)
            P_red_loss += P_red_loss_weight*P_red_loss_t
            if do_dopamine_mode:
                D_opa_signal[:, t] = P_red_loss_t.detach()

        # Jacobian penalty (/loss)]
        if mode == 'train' and do_jacobian_penalty and t in J_times:
            j_penalty = torch.zeros(batch_size).cuda()
            for l in J_layers:
              last_state = R_seq[l][t]
              prev_state = R_seq[l][t-1]
              j_penalty += jacobian_penalty(
                last_state, prev_state, mu).view(batch_size, -1).mean(axis=1)
            J_pen_loss += J_pen_loss_weight*j_penalty/len(J_layers)*(n_frames-t_start_loss)
            J_pen_signal[:, t] = j_penalty.detach().cpu()

    # Then supervised losses (which may be modulated by the unsupervised losses
    D_opa_signal /= D_opa_signal.sum(axis=1, keepdim=True)  # normalize every signal sequence over the time dimension
    for t in range(t_start_loss, n_frames):

        # Segmentation losses (supervised) S_seq is [batch_size, n_classes, w, h, n_frames]
        if do_segmentation:
            S_frame = S_seq[..., t]
            S_lbl_frame = S_lbl[..., t]
            if S_dic_loss_weight > 0:
                inter = (S_frame*S_lbl_frame).sum(axis=(-3, -2, -1))  # sum over n_classes, w and h
                union = (S_frame+S_lbl_frame).sum(axis=(-3, -2, -1))  # sum over n_classes, w and h
                S_dic_loss += D_opa_signal[:, t]*S_dic_loss_weight*(1.0 - (2*inter + 1.0)/(union + 1.0))
            if S_bce_loss_weight > 0:
                S_mse_loss += D_opa_signal[:, t]*S_mse_loss_weight*torch.square(S_frame - S_lbl_frame).mean(axis=(-3, -2, -1))
            if S_mse_loss_weight > 0:
                S_bce_loss += D_opa_signal[:, t]*S_bce_loss_weight*bce_loss_fn(S_frame, S_lbl_frame)

    # Total loss (weighted sum of above losses)
    P_red_loss = P_red_loss.mean()
    S_dic_loss = S_dic_loss.mean()
    S_bce_loss = S_bce_loss.mean()
    S_mse_loss = S_mse_loss.mean()
    J_pen_loss = J_pen_loss.mean()
    if mode == 'valid' and 0:
        plt.plot(range(t_start_loss, n_frames), D_opa_signal.mean(axis=0).detach().cpu()[t_start_loss:])
        plt.show()
    if mode == 'train':
        print(f'\rBatch ({batch_idx+1}/{n_batches}) - pred loss: {P_red_loss:.3f}, jaco loss: {J_pen_loss:.3f},'\
        # print(f'\rBatch ({batch_idx+1}/{n_batches}) - pred loss: {P_red_loss:.3f},'\
            + f' segm loss: [dic: {S_dic_loss:.3f}, bce: {S_bce_loss:.3f}, mse: {S_mse_loss:.3f}]', end='')
    return P_red_loss + S_bce_loss + S_mse_loss + S_dic_loss + J_pen_loss
    

def train_fn(train_dataloader, model, optimizer, epoch, plot_gif=True):

    model.train()
    plot_loss_train = 0.0
    n_batches = len(train_dataloader)   
    for i, (batch, S_lbl, _) in enumerate(train_dataloader):

        if speed_up_factor > 1:
              zero_frame = np.random.randint(speed_up_factor)
              batch = batch[..., zero_frame::speed_up_factor]
              S_lbl = S_lbl[..., zero_frame::speed_up_factor]
        if remove_ground:
              S_lbl = S_lbl[:, 1:, ...]
        E_seq, R_seq, P_seq, S_seq = model(batch)
        loss = loss_fn(E_seq, R_seq, S_seq, S_lbl, i, n_batches)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)  # slowdown?
        optimizer.step()
        plot_loss_train += loss.detach().item()/n_batches
        if i == 0 and plot_gif:
            plot_recons(batch, S_lbl, P_seq, S_seq, epoch=epoch)

    print('\r\nEpoch train loss : {}'.format(plot_loss_train))
    return plot_loss_train


def valid_fn(valid_dataloader, model, plot_gif=False):

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
            if plot_gif:  # used in test_net.py
                plot_recons(batch, S_lbl, P_seq, S_seq, sample_indexes=range(len(batch)), mode='test')
                exit()
            loss = loss_fn(E_seq, R_seq, S_seq, S_lbl, i, n_batches, mode='valid')
            plot_loss_valid += loss.detach().item()/n_batches
    
    print('Epoch validation loss: {}'.format(plot_loss_valid))
    return plot_loss_valid


def main():
    
    # Create or load model
    if not load_model:
        print(f'\nCreating model: {model_name}')
        model = SegNet(model_name, device,
            A_channels, R_channels, R_layers, S_layers,
            filter_sizes, n_classes, do_time_aligned,
            do_prediction, do_segmentation, do_jacobian_penalty)
        valid_losses, train_losses, last_epoch = [], [], 0
        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=learning_rate_schedule['milestones'],
            gamma=learning_rate_schedule['gamma'])
    else:
        print(f'\nLoading model: {model_name}')
        model, optimizer, scheduler, valid_losses, train_losses =\
            SegNet.load_model(model_name, epoch_to_load)
        last_epoch = scheduler.last_epoch
        
    # Load the dataset
    print('Loading dataset...')
    train_dataloader, valid_dataloader = get_datasets_seg(
        dataset_path=h5_path,
        tr_ratio=tr_ratio,
        n_samples=n_samples,
        batch_size_train=batch_size_train,
        batch_size_valid=batch_size_valid,
        occlusion=occlusion,
        augmentation=augmentation,
        dvs=dvs_mode)

    # Train and validate the network
    print('Training network...')
    for epoch in range(n_epochs_to_run):
        print(f'\nEpoch n°{last_epoch + epoch}')
        train_losses.append(train_fn(train_dataloader, model, optimizer, epoch))
        valid_losses.append(valid_fn(valid_dataloader, model))
        scheduler.step()
        if (last_epoch + epoch + 1) % n_epoch_save == 0:
            model.save_model(optimizer, scheduler, train_losses, valid_losses)


if __name__ == "__main__":
    main()
