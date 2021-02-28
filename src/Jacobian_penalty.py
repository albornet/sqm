import torch

def jacobian_penalty(state, prev_state, mu):
  jv_penalty = torch.tensor([1]).float().cuda()
  v = torch.ones_like(state)
  v.requires_grad = False
  jv_prod = torch.autograd.grad(
  	state,
  	prev_state,
  	grad_outputs=[v],
  	retain_graph=True,
  	create_graph=True,
  	allow_unused=True)[0]
  jv_penalty = (jv_prod - mu).clamp(min=0)**2
  return jv_penalty