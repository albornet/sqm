import os
import torch
import matplotlib.pyplot as plt
from src.Decoder import Decoder

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
decoder, opt, sched, T, A = Decoder.load_model(mother_name)
last_epoch = sched.last_epoch

fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Train loss', fontsize=14, color=color)
ax1.plot(range(last_epoch), T, color=color)
# ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel('Train accuracy', fontsize=14, color=color)  # we already handled the x-label with ax1
ax2.plot(range(last_epoch), A, color=color)
# ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()