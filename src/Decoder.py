import torch
import torch.nn as nn
import os

class Decoder(nn.Module):
  def __init__(self, input_size, hidden_sizes, decoder_name, device='cuda'):
    super(Decoder, self).__init__()
    self.input_size = input_size
    self.hidden_sizes = hidden_sizes
    self.n_hidden_sizes = len(hidden_sizes)
    self.decoder_name = decoder_name
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
          nn.Dropout(p=0.1),
          nn.Linear(hidden_sizes[-1], 2)))
    else:
      self.classifier = nn.Sequential(
        nn.Dropout(p=0.1),
        nn.Linear(input_size, 2))
    self.to(device)

  def forward(self, input_):
    input_ = input_.view(input_.size(0), -1)
    return self.classifier(input_)

  def save_model(self, optimizer, scheduler, train_losses, valid_losses):
    torch.save({
      'model_params': self.state_dict(),
      'device': self.device,
      'optimizer': optimizer,
      'scheduler': scheduler,
      'optimizer_params': optimizer.state_dict(),
      'scheduler_params': scheduler.state_dict(),
      'train_losses': train_losses,
      'valid_losses': valid_losses},
      f'./ckpt/{self.model_name}/{self.model_name}_ckpt_{scheduler.last_epoch:02}.pt')
    print('SAVED')

  @classmethod
  def load_model(cls, model_name, epoch_to_load=None):
    ckpt_dir = f'./ckpt/{model_name}/'
    ckpt_path = os.listdir(ckpt_dir)[-1]  # take last checkpoint (default)
    for ckpt in os.listdir(ckpt_dir):
      if str(epoch_to_load) in ckpt.split('_')[-1]:
        ckpt_path = ckpt
    save = torch.load(ckpt_dir + ckpt_path)
    model = cls(
      model_name=save['model_name'],
      device=save['device'])
    model.load_state_dict(save['model_params'])
    optimizer = save['optimizer']
    scheduler = save['scheduler']
    optimizer.load_state_dict(save['optimizer_params'])
    scheduler.load_state_dict(save['scheduler_params'])
    valid_losses = save['valid_losses']
    train_losses = save['train_losses']
    return model, optimizer, scheduler, train_losses, valid_losses
