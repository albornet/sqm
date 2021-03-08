import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transform(image_dims):
  transform = A.Compose([
      # A.Rotate(limit=(-10, 10), p=1.0),
      A.CenterCrop(height=image_dims[0], width=image_dims[1], p=1.0),
      # A.RandomBrightnessContrast(p=1.0),    
      # A.RandomGamma(p=1.0),
      A.GaussNoise(var_limit=(0.0, 50.0), mean=0.0, p=1.0),
      # A.Normalize((0.0,), (1.0,)),
      ToTensorV2()],
      additional_targets={f'image{t}': 'image' for t in range(1, 20)})
  return transform

def vernier_patch(image_dims, barH, barW, spaceH, offsetW, brightness, side):
  spaceW = offsetW - barW
  patch = np.zeros((2*barH + spaceH, 2*barW + spaceW))
  patch[:barH, :barW] = brightness
  patch[barH + spaceH:, barW + spaceW:] = brightness
  if side:
    return np.dstack([patch]*image_dims[-1])
  else:
    return np.dstack([np.fliplr(patch)]*image_dims[-1])

def place_patch(image_dims, frame, patch, row, col):
  patch_dims = patch.shape
  row -= patch_dims[0]//2
  col -= patch_dims[1]//2
  row0 = max(0, min(row, image_dims[0]))
  col0 = max(0, min(col, image_dims[1]))
  row1 = max(0, min(row + patch_dims[0], image_dims[0]))
  col1 = max(0, min(col + patch_dims[1], image_dims[1]))
  frame[row0:row1, col0:col1] += patch[row0-row:row1-row, col0-col:col1-col]

def choose_random_params(image_dims, n_frames, mode=None):
  offset = np.random.randint(1, 5)
  params1 = {
  'background_frames': 3, #np.random.randint(3, 6),
  'background_color': np.random.uniform(0.0, 0.3),
  'barH': np.random.randint(10, 20),
  'barW': np.random.randint(1, 2),
  'spaceH': np.random.randint(1, 3),
  'offsetW': np.array([offset]*n_frames),
  'side': np.array([np.random.randint(0, 2)]*n_frames),
  'brightness': np.random.uniform(0.7, 1.0),
  'row_0': int(np.random.normal(image_dims[0]/2, image_dims[0]/10)),
  'col_0': int(np.random.normal(image_dims[1]/2, image_dims[1]/10)),
  'd_row': 0,  # np.random.randint(0, 1),
  'd_col': offset//2 + 2}  # np.random.randint(1, 4)}
  params2 = {key: value for key, value in params1.items()}
  params2['d_col'] = -params1['d_col']
  if mode is not None:
    if 'V-' in mode:  # for tests
      offset = 1
      params1['background_frames'] = 3
      params1['background_color'] = 0.1
      params1['barH'] = 15
      params1['barW'] = 1
      params1['spaceH'] = 1
      params1['brightness'] = 0.9
      params1['row_0'] = image_dims[0]//2
      params1['col_0'] = image_dims[1]//2
      params1['d_row'] = 0
      params1['d_col'] = offset//2 + 2
      params2 = {key: value for key, value in params1.items()}
      params2['d_col'] = -params1['d_col']
    if mode == 'V' or mode == 'V-0':
      params1['offsetW'] = np.array([offset,] + [0]*(n_frames-1))
      params2['offsetW'] = np.array([offset,] + [0]*(n_frames-1))
    if 'V-AV' in mode or 'V-PV' in mode:
      APV_frame = int(mode[-1])
      side = np.random.randint(0, 2)
      APV_side = 1-side if 'V-AV' in mode else side
      params1['offsetW'] = np.array([offset,] + [0]*(APV_frame-1) + [offset,] + [0]*(n_frames-APV_frame-1))
      params1['side'] = np.array([side,] + [0]*(APV_frame-1) + [APV_side,] + [0]*(n_frames-APV_frame-1))
      params2['offsetW'] = np.array([offset,] + [0]*(n_frames-1))
      params2['side'] = np.array([side,] + [0]*(n_frames-1))
  return params1, params2

def draw_sequence(image_dims, n_frames, noise_level, mode=None):
  sequence = np.zeros((n_frames,) + image_dims)
  p1, p2 = choose_random_params(image_dims, n_frames, mode)
  background = p1['background_color']*np.ones(image_dims)
  for t in range(n_frames):
    frame = background*1
    for p in p1, p2:
      patch = vernier_patch(
        image_dims=image_dims, barH=p['barH'], barW=p['barW'], spaceH=p['spaceH'],
        offsetW=p['offsetW'][t], side=p['side'][t], brightness=p['brightness'])
      place_patch(image_dims, frame, patch, p['row_0'] + t*p['d_row'], p['col_0'] + t*p['d_col'])
    sequence[t] = frame
  sequence = np.roll(sequence, p1['background_frames'], axis=0)
  sequence[:p1['background_frames']] = p1['background_color']
  sequence = sequence.clip(max=p1['brightness'])
  if noise_level > 0.0:
    sequence += np.random.normal(0.0, noise_level, sequence.shape)
  return sequence, p1['side'][0]

def sqm_to_dvs(sequence, n_frames):
  dvs_sequence = 0.5*np.ones(sequence.shape)
  for t in range(1, n_frames):
    bigger = 0.5*(sequence[t] > sequence[t-1] + 0.1)
    smaller = 0.5*(sequence[t] < sequence[t-1] - 0.1)
    dvs_sequence[t] = dvs_sequence[t] - smaller + bigger
  return dvs_sequence

def create_epoch(n_samples, image_dims, n_frames, noise_level=0.0, do_dvs=False, do_transform=False, mode=None):

  t0 = time.time()
  final_image_dims = image_dims
  if do_transform:
    transform = get_transform(image_dims)
    image_dims = (image_dims[0]+20, image_dims[1]+20, image_dims[2])
  samples = np.zeros((n_samples, n_frames) + image_dims)
  labels = np.zeros((n_samples))
  for b in range(n_samples):
    sample, label = draw_sequence(image_dims, n_frames, noise_level, mode=mode)
    samples[b] = sample if do_dvs == False else sqm_to_dvs(sample, n_frames)
    labels[b] = label

  if do_transform:
    epoch_labels = torch.from_numpy(labels).long()
    epoch_samples = torch.zeros((n_samples, final_image_dims[-1], final_image_dims[0], final_image_dims[1], n_frames))
    samples = (255*samples).astype(np.uint8)
    for index in range(n_samples):
      sequence = samples[index]
      sequence0 = sequence[0]
      sampleT = {f'image{t}': sequence[t] for t in range(1, n_frames)}
      augment = transform(image=sequence0, **sampleT)
      sequence = [augment['image']] + [augment[f'image{t}'] for t in range(1, n_frames)]
      epoch_samples[index] = torch.stack(sequence, dim=3)
    return epoch_samples.cuda(), epoch_labels.cuda()

  else:
    samples = samples.transpose((0, 4, 2, 3, 1))  # for torch (channels after batch, frames last)
    return torch.from_numpy(samples).float().cuda(), torch.from_numpy(labels).long().cuda()

if __name__ == "__main__":
  epoch, labels = create_epoch(10, (64, 64, 3), 20, do_dvs=False, do_transform=False, mode='V-0')
  for t in range(20):
    plt.imshow(epoch[0, ..., t].cpu().permute((1,2,0)))
    plt.show()