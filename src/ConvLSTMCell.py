import torch
import torch.nn.functional as F
from torch import nn

class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super(ConvLSTMCell, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = (kernel_size-1)//2

        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        
    def forward(self, x, h, c):
        i_t = torch.sigmoid(self.Wxi(x) + self.Whi(h))
        f_t = torch.sigmoid(self.Wxf(x) + self.Whf(h))
        c_t = f_t*c + i_t*torch.tanh(self.Wxc(x) + self.Whc(h))
        o_t = torch.sigmoid(self.Wxo(x) + self.Who(h))
        h_t = o_t*torch.tanh(c_t)
        return h_t, c_t

# class hConvGRUCell(nn.Module):

#     def __init__(self, input_size, hidden_size, kernel_size=3):
#         super(hConvGRUCell, self).__init__()

#         self.padding = kernel_size//2
#         self.input_size = input_size
#         self.hidden_size = hidden_size
        
#         self.U1 = nn.Conv2d(hidden_size, hidden_size, 1)
#         self.U2 = nn.Conv2d(hidden_size, hidden_size, 1)
#         self.Wi = nn.Parameter(torch.empty(hidden_size, hidden_size, kernel_size, kernel_size))
#         self.We = nn.Parameter(torch.empty(hidden_size, hidden_size, kernel_size, kernel_size))
        
#         self.alpha = nn.Parameter(torch.empty((hidden_size, 1, 1)))
#         self.gamma = nn.Parameter(torch.empty((hidden_size, 1, 1)))
#         self.kappa = nn.Parameter(torch.empty((hidden_size, 1, 1)))
#         self.nu = nn.Parameter(torch.empty((hidden_size, 1, 1)))
#         self.mu = nn.Parameter(torch.empty((hidden_size, 1, 1)))
#         self.bn = nn.ModuleList([nn.BatchNorm2d(25, eps=1e-03, affine=True, track_running_stats=False) for i in range(4)])
#         self.activ = F.softplus

#         nn.init.orthogonal_(self.Wi)
#         nn.init.orthogonal_(self.We)      
#         nn.init.orthogonal_(self.U1.weight)
#         nn.init.orthogonal_(self.U2.weight)
#         for bn in self.bn:
#             nn.init.constant_(bn.weight, 0.1)      
#         nn.init.constant_(self.alpha, 0.1)
#         nn.init.constant_(self.gamma, 1.0)
#         nn.init.constant_(self.kappa, 0.5)
#         nn.init.constant_(self.nu, 0.5)
#         nn.init.constant_(self.mu, 1)
#         nn.init.uniform_(self.U1.bias.data, 1, 8.0 - 1)
#         self.U1.bias.data.log()
#         self.U2.bias.data = -self.U1.bias.data

#     def forward(self, input_, H2_prev):

#         G1 = torch.sigmoid((self.U1(H2_prev)))
#         C1 = self.bn[1](F.conv2d(H2_prev*G1, self.Wi, padding=self.padding))
#         H1 = self.activ(input_ - self.activ(C1*(self.alpha*H2_prev + self.mu)))
#         G2 = torch.sigmoid((self.U2(H1)))
#         C2 = self.bn[3](F.conv2d(H1, self.We, padding=self.padding))

#         # T2 = self.activ(self.kappa*H1 + C2*(self.nu*H1 + self.gamma)) # original
#         T2 = self.activ(self.kappa*H1 + self.activ(C2*(self.nu*H1 + self.gamma)))

#         H2_prev = (1 - G2)*H2_prev + G2*T2
#         H2_prev = self.activ(H2_prev)
#         return H2_prev, G2