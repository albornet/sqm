import torch
import torch.nn as nn
import os

class Decoder(nn.Module):
  def __init__(self, input_size, hidden_sizes, mother_name, device='cuda'):
    super(Decoder, self).__init__()
    self.input_size = input_size
    self.hidden_sizes = hidden_sizes
    self.n_hidden_sizes = len(hidden_sizes)
    self.mother_name = mother_name
    self.device = device
    if self.n_hidden_sizes > 0:
      self.classifier = nn.Sequential()
      hidden_sizes = [self.input_size] + self.hidden_sizes
      for i in range(self.n_hidden_sizes):
        self.classifier.add_module(
          f'hidden_{i}',
          nn.Sequential(
            nn.Linear(hidden_sizes[i], hidden_sizes[i+1]),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_sizes[i+1])))
      self.classifier.add_module(
        f'hidden_last',
        nn.Sequential(
          # nn.Dropout(p=0.1),
          nn.Linear(hidden_sizes[-1], 2)))
    else:
      self.classifier = nn.Sequential(
        # nn.Dropout(p=0.1),
        nn.Linear(input_size, 2))
    self.to(device)

  def forward(self, input_):
    input_ = input_.view(input_.size(0), -1)
    return self.classifier(input_)

  def save_model(self, optimizer, scheduler, train_losses, train_accurs):
    torch.save({
      'model_params': self.state_dict(),
      'input_size': self.input_size,
      'hidden_sizes': self.hidden_sizes,
      'device': self.device,
      'optimizer': optimizer,
      'scheduler': scheduler,
      'optimizer_params': optimizer.state_dict(),
      'scheduler_params': scheduler.state_dict(),
      'train_losses': train_losses,
      'train_accurs': train_accurs},
      f'./ckpt/{self.mother_name}/ckpt_decoder.pt')
    print('SAVED')

  @classmethod
  def load_model(cls, mother_name, epoch_to_load=None):
    ckpt_dir = f'./ckpt/{mother_name}/'
    list_dir = [c for c in os.listdir(ckpt_dir) if 'decoder' in c]
    ckpt_path = list_dir[-1]  # take last checkpoint (default)
    for ckpt in list_dir:
      if str(epoch_to_load) in ckpt.split('_')[-1]:
        ckpt_path = ckpt
    save = torch.load(ckpt_dir + ckpt_path)
    model = cls(
      input_size=save['input_size'],
      hidden_sizes=save['hidden_sizes'],
      mother_name=mother_name,
      device=save['device'])
    model.load_state_dict(save['model_params'])
    optimizer = save['optimizer']
    scheduler = save['scheduler']
    optimizer.load_state_dict(save['optimizer_params'])
    scheduler.load_state_dict(save['scheduler_params'])
    train_losses = save['train_losses']
    train_accurs = save['train_accurs']
    return model, optimizer, scheduler, train_losses, train_accurs
