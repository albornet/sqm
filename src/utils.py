import matplotlib.pyplot as plt
import numpy as np
import time

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
  offset = np.random.randint(1, 3)
  params1 = {
  'background_color': np.random.uniform(0.0, 0.3),
  'barH': np.random.randint(10, 20),
  'barW': np.random.randint(1, 2),
  'spaceH': np.random.randint(1, 3),
  'offsetW': np.array([offset]*n_frames),
  'side': np.array([np.random.randint(0, 2)]*n_frames),
  'brightness': np.random.uniform(0.7, 1.0),
  'row_0': int(np.random.normal(image_dims[0]/2, image_dims[0]/5)),
  'col_0': int(np.random.normal(image_dims[1]/2, image_dims[1]/5)),
  'd_row': np.random.randint(0, 1),
  'd_col': np.random.randint(1, 4)}
  params2 = {key: value for key, value in params1.items()}
  params2['d_col'] = -params1['d_col']
  if mode is not None:
    if mode == 'V':
      params1['offsetW'] = np.array([offset,] + [0]*n_frames-1)
      params2['offsetW'] = np.array([offset,] + [0]*n_frames-1)
    if 'V-AV' in mode or 'V-PV' in mode:
      APV_frame = int(mode[-1])
      APV_offset = -offset if 'V-AV' in mode else offset
      side = np.random.randint(0, 2)
      params1['offsetW'] = np.array([offset,] + [0]*(APV_frame-1) + [offset,] + [0]*(n_frames-APV_frame-1))
      params1['side'] = np.array([side,] + [0]*(APV_frame-1) + [APV_offset,] + [0]*(n_frames-APV_frame-1))
      params2['offsetW'] = np.array([offset,] + [0]*(n_frames-1))
      params2['side'] = np.array([side,] + [0]*(n_frames-1))
  return params1, params2

def draw_sequence(image_dims, n_frames, mode=None):
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
  return sequence.clip(max=1.0), p1['side'][0]

def create_epoch(n_samples, image_dims, n_frames, mode=None):
  samples = np.zeros((n_samples, n_frames) + image_dims)
  labels = np.zeros((n_samples))
  for b in range(n_samples):
    sample, label = draw_sequence(image_dims, n_framesmode=mode)
    samples[b] = sample
    labels[b] = label
  samples = samples.transpose((0, 4, 2, 3, 1))  # for torch (channels after batch, frames last)
  return torch.from_numpy(samples).cuda(), torch.from_numpy(labels).cuda()
