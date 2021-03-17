import os
import torch
import matplotlib.pyplot as plt
from src.Decoder import Decoder

# Mother and decoder net
mother_types = ['PredNet', 'PredNetTA', 'PredSegNetTA']
mother_dirs = ['DROPOUT_00_00', 'DROPOUT_01_01']
# mother_lab = ['p = 0.0', 'p = 0.2', 'p = 0.5']
threshold = 1.0
for mother_type in mother_types:
	fig, ax1 = plt.subplots()
	ls = ['-', '--']
	for i in range(2):
		mother_dir = mother_dirs[i]
		if mother_type == 'PredNet':
		  mother_name = 'PredNet_TA0_DM0_JP0-0_PR1-0_SM0-0_SB0-0_SD0-0_AC(3-16)_RC(16-64)_RL(h-h)_FS(5-5)_PL(1-1)_SL(0-0)'
		elif mother_type == 'PredNetTA':
		  mother_name = 'PredNet_TA1_DM0_JP0-0_PR1-0_SM0-0_SB0-0_SD0-0_AC(3-16)_RC(16-64)_RL(h-h)_FS(5-5)_PL(1-1)_SL(0-0)'
		elif mother_type == 'PredSegNetTA':
		  mother_name = 'PredNet_TA1_DM0_JP0-0_PR1-0_SM3-0_SB1-0_SD1-0_AC(3-16)_RC(16-64)_RL(h-h)_FS(5-5)_PL(1-1)_SL(0-1)'
		else:
		  print('Bad name, check the mother type or name?')
		  exit()
		mother_name = 'THRESH_' + str(int(threshold)) + '/' + mother_dir + '/' + mother_name
		decoder, opt, sched, T, A = Decoder.load_model(mother_name)
		last_epoch = 50  # sched.last_epoch

		color = 'tab:red'
		ax1.set_xlabel('Epoch', fontsize=16)
		ax1.set_ylabel('Train loss', fontsize=16, color=color)
		ax1.plot(range(last_epoch), T[:last_epoch], color=color, linestyle=ls[i])
		ax1.tick_params(axis='y', labelcolor=color, labelsize=12)
		ax1.set_ylim([0.00, 0.03])

		ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
		color = 'tab:blue'
		ax2.set_ylabel('Train accuracy', fontsize=16, color=color)  # we already handled the x-label with ax1
		ax2.plot(range(last_epoch), A[:last_epoch], color=color, linestyle=ls[i])
		ax2.axhline(y=0.5, xmin=0, xmax=last_epoch+1, color='gray', ls='--')
		ax2.tick_params(axis='y', labelcolor=color, labelsize=12)
		ax2.set_ylim([0.4, 1.0])
		ax2.tick_params(axis='x', labelsize=12)

	fig.tight_layout()  # otherwise the right y-label is slightly clipped
	plt.savefig(f'./ckpt/THRESH_{str(int(threshold))}/Training_decoder_{mother_type}', dpi=300)
	plt.show()