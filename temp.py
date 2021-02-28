import numpy as np
import matplotlib.pyplot as plt


n_layers = 3
TA = 1
I_seq = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
n_frames = len(I_seq)
E_seq = np.zeros((n_layers, TA + n_frames))
R_seq = np.zeros((n_layers, TA + n_frames))

def cell(x, y):
  return x + y

def conv(x):
  return x

def update(x):
  return x

# Run the network
for t in range(TA, n_frames+TA):
  A = I_seq[t-TA]

  # Top-down pass
  for l in reversed(range(n_layers)):
    E = E_seq[l][max(t-1, 0)]
    R = R_seq[l][t-TA]
    if l != n_layers - 1:
      R = cell(0, R_seq[l+1][t-TA])  # try with cell(0, R_seq[l+1][t-TA]) to understand
    else:
      R = cell(E, 0)
    R_seq[l][t] = R

  # Bottom-up pass
  for l in range(n_layers):
    A_hat = conv(R_seq[l][t-TA])
    E = A - A_hat
    E_seq[l][t] = E
    if l < n_layers - 1:
      A = update(E_seq[l][t-TA])

  # Plot stuff
  plt.subplot(2, 1, 1)
  plt.imshow(E_seq[:, TA:])
  plt.subplot(2, 1, 2)
  plt.imshow(R_seq[:, TA:])
  plt.show()
