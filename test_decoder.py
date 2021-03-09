# Mount drive folder (remove all this if not on drive!)
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from src.Decoder import Decoder
from src.SegNet import SegNet
from src.utils import create_epoch

# Training parameters
device = 'cuda'            # 'cuda' or 'cpu' ('cpu' never tested)
batch_size_test = 64
batches_per_epoch_test = 10
n_samples_per_epoch_test = batch_size_test*batches_per_epoch_test
decoded_layers = [1]
image_dims = (64, 64, 3)  # in torch actually: (batch-size, 3, 64, 64, n_frames)
n_frames = 20
noise_level = 0.04
do_transform = False
do_dvs = False
do_frame_concat = False
n_hidden_decoder = [64] if do_frame_concat else [512, 64]
learning_rate = 1e-6  # if do_frame_concat else 1e-5
plot_frames = False
loss_fn = nn.CrossEntropyLoss()  # nn.BCEWithLogitsLoss()

# Mother and decoder net
mother_dir = 'DROPOUT_01_01'  # 'DROPOUT_01_01', 'DROPOUT_02_05', 'DROUPOUT_00_00', '' (for BigNet)
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
print(f'Loading model and decoders at {mother_name}')
mother, _, _, _, _ = SegNet.load_model(mother_name)
decoder, _, _, _, _ = Decoder.load_model(mother_name)
mother.eval()
n_inputs = sum([mother.r_channels[:-1][l]*(64//(2**l))**2 for l in decoded_layers])
results = {
  'V-0':  {'means': [], 'stds': []},
  'V-PV': {'means': [], 'stds': []},
  'V-AV': {'means': [], 'stds': []}}

# Decoder training loop
for cond in ['V-0', 'V-PV', 'V-AV']:
  cond_frame = [0] if cond == 'V-0' else range(1, 15)
  for n in cond_frame:
    test_mode = cond if cond =='V-0' else cond + str(n)
    data, labels = create_epoch(n_samples_per_epoch_test, image_dims, n_frames,
      noise_level=noise_level, do_dvs=do_dvs, do_transform=do_transform, mode=test_mode)
    n_frames = data.shape[-1]
    n_correct = 0
    epoch_loss = 0.0
    accuracies = np.zeros(batches_per_epoch_test)
    for i in range(batches_per_epoch_test):

      # Take input and labels
      batch = data.narrow(dim=0, start=i*batch_size_test, length=batch_size_test)
      label = labels.narrow(dim=0, start=i*batch_size_test, length=batch_size_test)
      with torch.no_grad():
        E_seq, R_seq, P_seq, S_seq = mother(batch)

        # Prepare input; R_seq dim: [n_layers][n_frames][batch_sizes, r_channels, w, h]
        I_seq = [None for t in range(n_frames)]
        for t in range(n_frames):
          I_seq[t] = torch.stack(
            [R_seq[l][t].view(batch_size_test, -1) for l in decoded_layers], dim=1)

        # Run decoder
        if do_frame_concat:
          input_ = torch.cat([I_seq[t] for t in range(n_frames)], dim=1)
          decoding = decoder(input_)
        else:
          decoding = torch.zeros((batch_size_test, 2)).cuda()
          zero_evd = torch.zeros((batch_size_test, 2)).cuda()
          confides = torch.zeros((batch_size_test, n_frames))
          # for t in range(3, n_frames):
          #   decoding_t = decoder(I_seq[t])
          #   decoding += decoding_t
          for t in range(3, n_frames):
            decoding_t = decoder(I_seq[t])
            confidence = torch.abs(decoding[:, 1] - decoding[:, 0])  # decoding.detach?
            decoding += torch.where(torch.stack([confidence]*2, dim=1) < 1.0, decoding_t, zero_evd)
            confides[:, t] = confidence

        # Get performance
        loss = loss_fn(decoding, label)
        epoch_loss += loss.detach().item()
        predictions = torch.argmax(decoding, dim=1)
        n_correct = (predictions.long()==label).sum()
        accuracy = n_correct/batch_size_test
        accuracies[i] = accuracy

        # Internal plot if needed
        if plot_frames and i==0:
          for t in range(n_frames):
            plt.subplot(1, 3, 1)
            plt.imshow(batch[0, ..., t].cpu().permute((1,2,0)))
            plt.title(label[0])
            plt.subplot(1, 3, 2)
            plt.imshow(P_seq[0, ..., t].cpu().permute((1,2,0)))
            plt.subplot(1, 3, 3)
            plt.imshow(S_seq[0, ..., t].cpu().permute((1,2,0)))
            plt.show()

    # Store performance
    epoch_loss /= (n_samples_per_epoch_test)
    results[cond]['means'].append(accuracies.mean())
    results[cond]['stds'].append(accuracies.std()/np.sqrt(batches_per_epoch_test))  # stderr
    print(f'Cond {test_mode} accuracy: {accuracies.mean()}, loss: {epoch_loss}')

# Final plot
fig, ax = plt.subplots()
for cond in ['V-PV', 'V-AV']:
  ax.errorbar(range(1, 15), results[cond]['means'], yerr=results[cond]['stds'], fmt='-o')
  ax.set_xlabel('2nd offset frame')
  ax.set_ylabel('Accuracy')
plt.hlines(results['V-0']['means'][0], 0, 15, colors='gray', linestyles='dashed')
plt.savefig('./ckpt/' + mother_name + '/sqm_results', dpi=300)
plt.show()
