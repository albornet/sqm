import os
import torch
import matplotlib.pyplot as plt

# Network parameters
model_name = 'PredNet_TA1_JP0-0_PR1-0_SM0-0_SB0-0_SD0-0_CB0-0_CD0-0_AC(3-16-32)_RC(16-32-64)_RL(h-h-h)_FS(3-3-3)_PL(1-1-1)_SL(0-0-0)_CL(0-0-0)'
ckpt_dir = f'./ckpt/{model_name}/'
ckpt_files = os.listdir(ckpt_dir)
ckpt_path = ckpt_dir + [f for f in ckpt_files if '.pt' in f][-1]

epoch_load = int(ckpt_path.split('.pt')[0].split('_')[-1])
checkpoint = torch.load(ckpt_path)
T, V = checkpoint['train_losses'], checkpoint['valid_losses']
plt.plot(list(range(epoch_load)), V, label='valid')
plt.plot(list(range(epoch_load)), T, label='train')
# axes = plt.gca()
# axes.set_ylim([0.005,0.01])
plt.legend()
plt.show()