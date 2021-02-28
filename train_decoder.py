# Mount drive folder (remove all this if not on drive!)
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from src.Decoder import Decoder
from src.SegNet import SegNet
from src.utils import create_epoch

# Training parameters
device = 'cuda'            # 'cuda' or 'cpu' ('cpu' never tested)
load_model = False         # if False, create a new networks
epoch_to_load = None       # None for last epoch; not used if load_model == False
n_epochs_to_run = 100      # from the last epoch if load_model == True
n_epoch_save = 5           # will save a new checkpoint every n_epoch_save
learning_rate = 1e-5       # is modified by the scheduler
learning_rate_schedule = {'milestones': list(range(0, 2*n_epochs_to_run, 10)), 'gamma': 0.5}
batch_size_train = 16      # try larger and larger values, until it does not fit
batch_size_valid = 32      # faster because no need to loss.backward()
batches_per_epoch_train = 30
batches_per_epoch_valid = 2
n_samples_per_epoch_train = batch_size_train*batches_per_epoch_valid
n_samples_per_epoch_train = batch_size_train*batches_per_epoch_train
decoded_layers = [0]
n_hidden_decoder = [1024, 256, 64]
image_dims = (64, 64, 3)  # in torch actually: (batch-size, 3, 64, 64, n_frames)
n_frames = 20
train_mode = 'V'

# Mother and decoder net
mother_name = 'PredNet_TA0_JP0-0_PR1-0_SM0-0_SB0-0_SD0-0_AC(3-32)_RC(32-64)_RL(h-h)_FS(3-3)_PL(1-1)_SL(0-0)'
mother, _, _, _, _ = SegNet.load_model(mother_name)
n_inputs = sum([mother.r_channels[:-1][l]*(64//(2**l))**2 for l in decoded_layers])
decoder_name = f'Decoder_NI{n_inputs}'
decoder = Decoder(n_inputs, n_hidden_decoder, decoder_name, device)
optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()  # nn.BCEWithLogitsLoss()

# Decoder training loop
epoch_losses = []
for i in range(n_epochs_to_run):
  batch_losses = []
  data, labels = create_epoch(n_samples_per_epoch_train, image_dims, n_frames, train_mode)
  n_frames = data.shape[-1]

  n_correct = 0
  epoch_loss = 0.0
  for i in range(batches_per_epoch_train):

    batch = data.narrow(dim=0, start=i*batch_size_train, length=batch_size_train)
    label = labels.narrow(dim=0, start=i*batch_size_train, length=batch_size_train)
    with torch.no_grad():
      E_seq, R_seq, P_seq, S_seq = mother(batch)

    # R_seq dim: [n_layers][n_frames][batch_sizes, r_channels, w, h]
    input_ = [None for t in range(n_frames)]
    for t in range(n_frames):
      input_[t] = torch.stack(
        [R_seq[l][t].view(batch_size_train, -1) for l in decoded_layers], dim=1)

    frame_to_decode = 10
    D_seq = torch.zeros(batch_size_train, 2, n_frames).cuda()
    for t in range(n_frames):
      D_seq[..., t] = decoder(input_[t])
    # decoding = D_seq[..., :].mean(axis=-1)  # .squeeze()
    decoding = D_seq[..., 10]

    loss = loss_fn(decoding, label)
    loss.backward()
    optimizer.step()
    epoch_loss += loss.detach().item()
    decoding = torch.argmax(decoding.detach(),dim=1)    
    n_correct += (decoding.long()==label).sum()

    plot_predictions = False
    if plot_predictions:
      for t in range(n_frames):
        plt.imshow(S_seq[0, ..., t].detach().cpu().permute((1,2,0)))
        plt.show()

    plot_input = True
    if plot_input:
      for t in range(n_frames):
        plt.imshow(batch[0, ..., t].detach().cpu().permute((1,2,0)))
        plt.title(label[0].detach().cpu())
        plt.show()

  epoch_loss /= (n_samples_per_epoch_train)
  epoch_losses.append(epoch_loss)
  accuracy = n_correct/(batch_size_train*batches_per_epoch_train)
  print(f'Epoch accuracy: {accuracy}, epoch loss: {epoch_loss}')
