import numpy as np
import imageio
import torch

def SQM(SQM_params, n_steps = 20, im_dim = (3, 64, 64), vernier_dimensions = 'default', empty_frames_start = 2,  position=None):
  # produces any SQM sequence, given a parameter dictionnary

  if vernier_dimensions == 'default':
    # default param
    bar_w, bar_h = (1, 10)
    h_space, v_space = (1,1)
    frame_offset = 2
  elif vernier_dimensions == 'random':
    for p in random_Vparam():
      # generate random parameters
      break
    bar_w,bar_h,h_space,v_space,_, frame_offset = p

  elif type(vernier_dimensions) == type(()) and len(vernier_dimensions) == 6:
    #given param tuple
    bar_w,bar_h,h_space,v_space,_, frame_offset = vernier_dimensions
  else:
    assert False, 'vernier dimensions not clear'
     
  if position == None:
    # centered
    w, h =  (2*bar_w + h_space, 2*bar_h+ v_space)
    midx, midy = tuple([i//2 for i in im_dim[1:]])
    x0, y0 = midx - w//2, midy - h//2
  else:
    x0, y0 = position
    
  sequence = torch.zeros((im_dim[0], im_dim[1], im_dim[2], n_steps))
  img0 = torch.zeros(im_dim)

  for t in range(empty_frames_start):
    sequence[:,:,:,t] = img0*1

  b = frame_offset
  if SQM_params['stream'] in ['left','l', 'L']:
    b *= -1

  for t_in_seq in range(empty_frames_start, n_steps):
    t_effective = t_in_seq - empty_frames_start
    img = img0*1
    
    # static stream
    if t_effective > 0:
      patch = Vernier(bar_w, bar_h, h_space, v_space, sense = 0, n_channels = im_dim[0])
      img = addpatch(img, patch, (y0, x0-b*t_effective))
    
    # modified stream
    sense = SQM_params[t_effective] if t_effective in SQM_params else 0
    patch = Vernier(bar_w, bar_h, h_space, v_space, sense = sense, n_channels = im_dim[0])
    img = addpatch(img, patch, (y0, x0+b*t_effective))
    sequence[:,:,:,t_in_seq] = img

  return sequence

def SQM_train(dimensions,n_steps = 20, im_dim = (3, 64,64), empty_frames_start = 2, scalemax = 1.):
  # generates random SQM sequences designed for training (with offset only in initial frame)
  # build param dictionnary:
  bar_w,bar_h,h_space,v_space,sense, frame_offset = dimensions
  param_dic = {0: sense, 'stream':None} # stream does not matter, but is here to avoid bug
  # generate initial position:
  patch_w, patch_h =  (2*bar_w + h_space, 2*bar_h+ v_space)
  margin = 4
  ypos = bounded_normal_int(margin, im_dim[1]-patch_h-margin, 1)
  xpos = bounded_normal_int(patch_w+margin, im_dim[2]-2*patch_w-margin, 1)
  return SQM(param_dic, n_steps = n_steps, im_dim = im_dim, \
   vernier_dimensions = dimensions, empty_frames_start = empty_frames_start,  position=(xpos,ypos))

def sqm_train_epoch(n_samples, scalemax = 1., device='cuda'):
  # returns a batch of size n_samples filled with training examples
  param_list = list(random_Vparam(n_samples))
  batch = [SQM_train(param, scalemax = scalemax) for param in param_list]
  label = [(param[4]+1)*.5 for param in param_list]
  return torch.stack(batch, dim=0).to(device), torch.tensor(label).long().to(device)

def get_all_SQM(init_sense, stream, frames=[1,4,8], max_n_step = 20):
  # returns classical V-AV and V_PV sqm examples, param and seq (not for training)
  verniers = [-1, 1]
  assert init_sense in [1,0,-1] and stream in ['R', 'L'], 'wrong args'
  param_list = [('V', {'stream':stream, 0:init_sense})]
  for v in verniers:
    for f in frames:
      name = 'V' if init_sense else '0'     
      if v*init_sense > 0:
        name += '-PV'
      elif v*init_sense < 0:
        name += '-AV'
      name += str(f)
      param = {'stream' : stream, 0: init_sense, f: v}
      param_list.append((name, param))
  sqm_list = [SQM(p[1], n_steps = max_n_step) for p in param_list]
  return sqm_list, param_list

#### internal functions

def Vernier(bar_w, bar_h, h_space, v_space =1, sense=0, n_channels = 3):
  #Vernier-drawing function.
  h_offset = h_space - bar_w
  patch_dim = (n_channels, 2*bar_w + h_offset, 2*bar_h+ v_space)
  patch = np.zeros(patch_dim)
  if sense == 0:
    startx = patch_dim[1]//2 - bar_w//2
    stopx = startx + bar_w
    starty = patch_dim[2]//2 - v_space//2
    stopy = starty + v_space
    patch[:,startx:stopx, :starty] = 1 
    patch[:,startx:stopx, stopy:] = 1 #weird stuff on y-axis due to .T "hack" in return stat.
  elif sense in [-1,1]:
    patch[:,:bar_w,:bar_h] = 1
    patch += patch[:,::-1,::-1]
    patch = patch [:,::sense, :]
    patch = patch
  else:
    assert False, 'sense not understood'
  
  return patch.transpose(0,2,1)

def Vernier_short(dimensions):
  # same as Vernier(), just made to shorten calls
  w,h,hs,vs,sense,_ = dimensions
  return Vernier(w,h,hs,vs,sense)


def random_Vparam(batch_size = 1, no_0_sense = True):
  # Random parameter generating function Value ranges are:
  # bar_w = 1:3, bar_h = 7:15, h_space = 1:5, v_space = 1, frame offset = 1:4
  if no_0_sense:
    senses = (0,2) # 0 or 1
  else:
    senses = (-1,2)  # -1 or 0 or 1
  params = []
  for minmax in [(1,4),(7,16),(1,6),(1,2),senses, (1, 5)]:
    # in order : bar_w/bar_h/h_space/v_space/sense/frame_offset(speed)
    params.append(np.random.randint(minmax[0],minmax[1],batch_size))
  if no_0_sense:
    params[4] = params[4]*2 -1 # change val : 0/1 to -1/1
  return zip(params[0],params[1],params[2],params[3],params[4], params[5])


def addpatch(img0, patch, pos):
  # adds patch to bigger grid
  # WARNING : - operation is +=, followed by thresholding (odd override behaviour)
  #           - RGB channels first
  img = img0 *1
  im_dim = img.shape
  patch_dim = patch.shape
  x0 = max(0, min(pos[0], im_dim[1]))
  y0 = max(0, min(pos[1], im_dim[2]))
  x1 = max(0,min(pos[0] + patch_dim[1], im_dim[1]))
  y1 = max(0,min(pos[1] + patch_dim[2], im_dim[2]))
  px0 = x0 - pos[0]
  px1 = x1 - pos[0]
  py0 = y0 - pos[1]
  py1 = y1 - pos[1]
  img[:,x0:x1,y0:y1] += patch[:,px0:px1,py0:py1]
  img[img>1] = 1
  return img


def gifify(sequence, name = 'SQM'):
  #takes a (n_channels ,w, h, n_frames) tensor and makes a gif
  sequence = sequence.cpu().numpy().transpose(-1,1,2,0)
  gif_frames = [s.astype(np.uint8) *255 for s in sequence]
  imageio.mimsave(f'./gif/{name}.gif', gif_frames, duration=0.15)


def bounded_normal_int(mini, maxi, batch_size):
  # draws number from normal distribution and yields closest int (for position)
  normal0to1 = np.random.normal(0,.33,batch_size)*.5+.5
  ret = (maxi*normal0to1+mini).astype(int)
  if batch_size == 1:
    ret = ret[0]
  return ret


def rgb_to_dvs(batch_seq):
  # take a batch of sequence and compute corresponding dvs input
  batch_seq.fill(0)
  batch_size = batch_seq.shape[0]
  batch_diff = np.diff(batch_seq.sum(axis=1), n=1, axis=-1, prepend=0)
  for b in range(batch_size):
    batch_seq[b, 0:2, batch_diff[b] > 0] = 255
    batch_seq[b, 1, batch_diff[b] < 0] = 255
