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
n_epoch_save = 5          # will save a new checkpoint every n_epoch_save
learning_rate_schedule = {'milestones': list(range(0, 2*n_epochs_to_run, 50)), 'gamma': 0.5}
batch_size_train = 32      # try larger and larger values, until it does not fit
batch_size_valid = 32
batches_per_epoch_train = 50
batches_per_epoch_valid = 5
n_samples_per_epoch_train = batch_size_train*batches_per_epoch_train
n_samples_per_epoch_valid = batch_size_valid*batches_per_epoch_valid
decoded_layers = [1]
image_dims = (64, 64, 3)  # in torch actually: (batch-size, 3, 64, 64, n_frames)
n_frames = 20
train_mode = 'V'
noise_level = 0.0
do_transform = False
do_dvs = False
do_frame_concat = False
n_hidden_decoder = [64] if do_frame_concat else [512, 64]
learning_rate = 1e-6  # if do_frame_concat else 1e-5

# Mother and decoder net
mother_dir = 'DROPOUT_02_05'  # 'DROPOUT_01_01', 'DROPOUT_02_05', 'DROUPOUT_00_00', '' (for BigNet)
mother_type = 'PredNetTA'  # 'PredNet', 'PredNetTA', 'PredSegNetTA'
if mother_type == 'PredNet':
  mother_name = 'PredNet_TA0_DM0_JP0-0_PR1-0_SM0-0_SB0-0_SD0-0_AC(3-16)_RC(16-64)_RL(h-h)_FS(5-5)_PL(1-1)_SL(0-0)'
elif mother_type == 'PredNetTA':
  mother_name = 'PredNet_TA1_DM0_JP0-0_PR1-0_SM0-0_SB0-0_SD0-0_AC(3-16)_RC(16-64)_RL(h-h)_FS(5-5)_PL(1-1)_SL(0-0)'
elif mother_type == 'PredSegNetTA':
  mother_name = 'PredNet_TA1_DM0_JP0-0_PR1-0_SM3-0_SB1-0_SD1-0_AC(3-16)_RC(16-64)_RL(h-h)_FS(5-5)_PL(1-1)_SL(0-1)'
elif mother_type == 'BigNet':
  mother_name = ''
else:
  print('Bad name, check the mother type or name?')
  exit()
mother_name = mother_dir + '/' + mother_name
mother, _, _, _, _ = SegNet.load_model(mother_name)
mother.eval()
n_inputs = sum([mother.r_channels[:-1][l]*(64//(2**l))**2 for l in decoded_layers])
if do_frame_concat:
  n_inputs *= n_frames
decoder = Decoder(n_inputs, n_hidden_decoder, mother_name, device)
optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=list(range(0, 2*n_epochs_to_run, n_epoch_save)), gamma=0.5)
loss_fn = nn.CrossEntropyLoss()  # nn.BCEWithLogitsLoss()

# Decoder training loop
epoch_losses = []
epoch_accurs = []
for e in range(n_epochs_to_run):
  decoder.train()
  batch_losses = []
  data, labels = create_epoch(n_samples_per_epoch_train, image_dims, n_frames,
    noise_level=noise_level, do_dvs=do_dvs, do_transform=do_transform, mode=train_mode)
  n_frames = data.shape[-1]

  n_correct = 0
  epoch_loss = 0.0
  for i in range(batches_per_epoch_train):

    # Take input and labels
    batch = data.narrow(dim=0, start=i*batch_size_train, length=batch_size_train)
    label = labels.narrow(dim=0, start=i*batch_size_train, length=batch_size_train)
    with torch.no_grad():
      E_seq, R_seq, P_seq, S_seq = mother(batch)

    # Prepare input; R_seq dim: [n_layers][n_frames][batch_sizes, r_channels, w, h]
    I_seq = [None for t in range(n_frames)]
    for t in range(n_frames):
      I_seq[t] = torch.stack(
        [R_seq[l][t].view(batch_size_train, -1) for l in decoded_layers], dim=1)
 
    # Run decoder
    if do_frame_concat:
      input_ = torch.cat([I_seq[t] for t in range(n_frames)], dim=1)
      decoding = decoder(input_)
    else:
      decoding = torch.zeros((batch_size_train, 2)).cuda()
      zero_evd = torch.zeros((batch_size_train, 2)).cuda()
      confides = torch.zeros((batch_size_train, n_frames))
      # for t in range(3, n_frames):
      #   decoding_t = decoder(I_seq[t])
      #   decoding += decoding_t
      for t in range(3, n_frames):
        decoding_t = decoder(I_seq[t])
        confidence = torch.abs(decoding[:, 1] - decoding[:, 0])  # decoding.detach?
        decoding += torch.where(torch.stack([confidence]*2, dim=1) < 1.0, decoding_t, zero_evd)
        confides[:, t] = confidence

    # Backpropagation
    loss = loss_fn(decoding, label)
    loss.backward()
    optimizer.step()
    epoch_loss += loss.detach().item()
    prediction = torch.argmax(decoding.detach(), dim=1)
    n_correct += (prediction.long()==label).sum()

    plot = False
    if plot:
      for t in range(n_frames):
        plt.subplot(1, 4, 1)
        plt.imshow(batch[0, ..., t].detach().cpu().permute((1,2,0)))
        plt.title(label[0].detach().cpu())
        plt.subplot(1, 4, 2)
        plt.imshow(P_seq[0, ..., t].detach().cpu().permute((1,2,0)))
        plt.subplot(1, 4, 3)
        plt.imshow(S_seq[0, ..., t].detach().cpu().permute((1,2,0)))
        plt.subplot(1, 4, 4)
        plt.plot(range(n_frames), confides.detach().cpu()[0])
        plt.show()

  scheduler.step()
  epoch_loss /= (n_samples_per_epoch_train)
  epoch_accur = n_correct/n_samples_per_epoch_train
  epoch_losses.append(epoch_loss)
  epoch_accurs.append(epoch_accur)
  print(f'Epoch {e} train accuracy: {epoch_accur}, train loss: {epoch_loss}')
  if (e + 1) % n_epoch_save == 0:
      decoder.save_model(optimizer, scheduler, epoch_losses, epoch_accurs)
  
  # # Small stupid validation
  # decoder.eval()
  # valid_batch, valid_labels = create_epoch(n_samples_per_epoch_valid, image_dims,
  #   n_frames, do_dvs=do_dvs, do_transform=do_transform, mode=train_mode)
  # with torch.no_grad():
  #   n_correct = 0
  #   for i in range(batches_per_epoch_valid): 
  #     vbatch = data.narrow(dim=0, start=i*batch_size_valid, length=batch_size_valid)
  #     vlabel = labels.narrow(dim=0, start=i*batch_size_valid, length=batch_size_valid)
  #     E_seq, R_seq, P_seq, S_seq = mother(vbatch)
  #     I_seq = [None for t in range(n_frames)]
  #     for t in range(n_frames):
  #       I_seq[t] = torch.stack(
  #         [R_seq[l][t].view(batch_size_valid, -1) for l in decoded_layers], dim=1)
  #     if do_frame_concat:
  #       input_ = torch.cat([I_seq[t] for t in range(n_frames)], dim=1)
  #       decoding = decoder(input_)
  #     else:
  #       decoding = torch.zeros((batch_size_valid, 2)).cuda()
  #       for t in range(3, n_frames):
  #         decoding += decoder(I_seq[t])/(n_frames-3)
  #     prediction = torch.argmax(decoding.detach(),dim=1)
  #     n_correct += (prediction.long()==vlabel).sum()
  #   accuracy = n_correct/n_samples_per_epoch_valid
  #   print(f'\tvalid accuracy: {accuracy}')