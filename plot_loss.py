import os
import torch
import matplotlib.pyplot as plt
from src.SegNet import SegNet

mother_dirs = ['DROPOUT_00_00', 'DROPOUT_01_01']
threshold = 1.0
ls = ['-', '--']
lb = [', p = 0.0', ', p = 0.1']
for mother_type in ['PredNet', 'PredNetTA', 'PredSegNetTA']:
	fig, ax1 = plt.subplots()
	for i in range(2):
		mother_dir = mother_dirs[i]
		if mother_type == 'PredNet':
		  mother_name = 'PredNet_TA0_DM0_JP0-0_PR1-0_SM0-0_SB0-0_SD0-0_AC(3-16)_RC(16-64)_RL(h-h)_FS(5-5)_PL(1-1)_SL(0-0)'
		elif mother_type == 'PredNetLSTM':
		  mother_name = 'PredNet_TA0_DM0_JP0-0_PR1-0_SM0-0_SB0-0_SD0-0_AC(3-16)_RC(16-64)_RL(l-l)_FS(5-5)_PL(1-1)_SL(0-0)'
		elif mother_type == 'PredNetTA':
		  mother_name = 'PredNet_TA1_DM0_JP0-0_PR1-0_SM0-0_SB0-0_SD0-0_AC(3-16)_RC(16-64)_RL(h-h)_FS(5-5)_PL(1-1)_SL(0-0)'
		elif mother_type == 'PredSegNetTA':
		  mother_name = 'PredNet_TA1_DM0_JP0-0_PR1-0_SM3-0_SB1-0_SD1-0_AC(3-16)_RC(16-64)_RL(h-h)_FS(5-5)_PL(1-1)_SL(0-1)'
		elif mother_type == 'BigNet':
		  mother_name = ''
		else:
		  print('Bad name, check the mother type or name?')
		  exit()
		mother_name = 'THRESH_' + str(int(threshold)) + '/' + mother_dir + '/' + mother_name
		mother, opt, sched, T, V = SegNet.load_model(mother_name)
		last_epoch = 20
		ax1.plot(range(last_epoch), T[:last_epoch], color='tab:red', label='Train loss' + lb[i], linestyle=ls[i])
		ax1.plot(range(last_epoch), V[:last_epoch], color='tab:blue', label='Valid loss' + lb[i], linestyle=ls[i])
	ax1.set_xlabel('Epoch', fontsize=16)
	ax1.set_ylabel('Loss', fontsize=16)
	ax1.tick_params(axis='y', labelsize=12)
	if mother_type in ['PredNet', 'PredNetTA']:
		plt.ylim(0.09, 0.69)
	plt.locator_params(axis='x', nbins=last_epoch)
	plt.legend(fontsize=16)
	plt.savefig(f'./ckpt/THRESH_{str(int(threshold))}/Training_network_{mother_dir}_{mother_type}', dpi=300)
	plt.show()
