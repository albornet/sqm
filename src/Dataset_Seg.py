import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
import numpy as np
import h5py
import albumentations as A
from albumentations.pytorch import ToTensorV2

def add_bar(input_):
  batch_size, n_channels, height, width, n_frames = input_.shape
  poses = np.random.randint(width, size=batch_size)
  sizes = np.random.randint(width//10, size=batch_size)
  add = np.random.choice([0, 1], p=[0.5, 0.5], size=batch_size)
  for i, (p, s, a) in enumerate(zip(poses, sizes, add)):
    if a:
      input_[i, :, :, p:p+s, :] = 0

transform = A.Compose([
    A.OneOf([
        A.RandomSizedCrop(min_max_height=(50, 63), height=64, width=64, p=0.5),
        A.PadIfNeeded(min_height=64, min_width=64, p=0.5)], p=1),    
    A.HorizontalFlip(p=0.5),              
    A.Rotate(limit=(-10, 10), p=1.0),
    A.RandomBrightnessContrast(p=1.0),    
    A.RandomGamma(p=1.0),
    A.GaussNoise(var_limit=(0.0, 50.0), mean=0.0, p=1.0),
    A.Normalize((0.0,), (1.0,)),
    ToTensorV2()],
    additional_targets={f'image{t}': 'image' for t in range(1, 20)})

class H5Dataset_Seg(data.Dataset):

  def __init__(self, h5_path, from_to, occlusion, augmentation, dvs):
    super(H5Dataset_Seg, self).__init__()
    self.augmentation = augmentation
    sample_dataset = 'dvs_samples' if dvs else 'rgb_samples'
    with h5py.File(h5_path, 'r') as f:  # sample dims for pytorch: (batch_id, channel, row, column, frame_id)
      self.img_samples = np.array(f.get(sample_dataset))[from_to[0]: from_to[1]] #.transpose(0,-1,2,3,1)
      self.lbl_segment = np.array(f.get('lbl_grouping'))[from_to[0]: from_to[1]] #.transpose(0,-1,2,3,1) #[:, 1:, ...]
      self.lbl_classif = np.array(f.get('lbl_visibles'))[from_to[0]: from_to[1]]
      if occlusion:
        for n in range(10):
          add_bar(self.img_samples)
      if not self.augmentation:
        self.img_samples = self.img_samples.transpose(0,-1,2,3,1)
        self.lbl_segment = self.lbl_segment.transpose(0,-1,2,3,1)
        
  def __getitem__(self, index):
    lbl_class = torch.from_numpy(self.lbl_classif[index]).float()
    if self.augmentation:
      n_frames = self.img_samples.shape[1]
      samples = self.img_samples[index]
      lbl_segm = self.lbl_segment[index]
      sample0 = samples[0]
      sampleT = {f'image{t}': samples[t] for t in range(1, n_frames)}
      lbl_segm  = [lbl_segm[t] for t in range(n_frames)]
      augment = transform(image=sample0, masks=lbl_segm, **sampleT)
      samples = [augment['image']] + [augment[f'image{t}'] for t in range(1, n_frames)]
      samples = torch.stack(samples, dim=3)
      lbl_segm = [torch.from_numpy(augment['masks'][t]) for t in range(n_frames)]
      lbl_segm = torch.stack(lbl_segm, dim=3).permute((2, 0, 1, 3)).float()
    else:
      samples = torch.from_numpy(self.img_samples[index])/255.0
      lbl_segm  = torch.from_numpy(self.lbl_segment[index]).float()
    return samples.to(device='cuda'), lbl_segm.to(device='cuda'), lbl_class.to(device='cuda')
    
  def __len__(self):
    return self.img_samples.shape[0]

def get_datasets_seg(dataset_path, tr_ratio, n_samples, batch_size_train, batch_size_valid, 
  occlusion=False, augmentation=False, dvs=False, mode=None):
  
  if mode != 'test':
    train_bounds     = (0, int(n_samples*tr_ratio))
    train_dataset    = H5Dataset_Seg(dataset_path, train_bounds, occlusion, augmentation, dvs)
    train_dataloader = data.DataLoader(train_dataset,
      batch_size=batch_size_train, shuffle=True, sampler=None, batch_sampler=None, num_workers=0, collate_fn=None,
      pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None, prefetch_factor=2, persistent_workers=False)
  else:
    train_dataloader = None
  
  valid_bounds     = (int(n_samples*tr_ratio), None)  # None means very last one
  valid_dataset    = H5Dataset_Seg(dataset_path, valid_bounds, occlusion, augmentation, dvs)
  valid_dataloader = data.DataLoader(valid_dataset,
    batch_size=batch_size_valid, shuffle=True, sampler=None, batch_sampler=None, num_workers=0, collate_fn=None,
    pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None, prefetch_factor=2, persistent_workers=False)
  
  return train_dataloader, valid_dataloader
