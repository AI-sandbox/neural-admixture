import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class _L0Norm(nn.Module):
    def __init__(self, origin, loc_mean=0, loc_sdev=0.01, beta=2 / 3, gamma=-0.1,
                 zeta=1.1, fix_temp=True):
        """
        Base class of layers using L0 Norm
        :param origin: original layer such as nn.Linear(..), nn.Conv2d(..)
        :param loc_mean: mean of the normal distribution which generates initial location parameters
        :param loc_sdev: standard deviation of the normal distribution which generates initial location parameters
        :param beta: initial temperature parameter
        :param gamma: lower bound of "stretched" s
        :param zeta: upper bound of "stretched" s
        :param fix_temp: True if temperature is fixed
        """
        super(_L0Norm, self).__init__()
        self._origin = origin
        self._size = self._origin.weight.size()
        self.loc = nn.Parameter(torch.zeros(self._size).normal_(loc_mean, loc_sdev))
        self.temp = beta if fix_temp else nn.Parameter(torch.zeros(1).fill_(beta))
        self.register_buffer("uniform", torch.zeros(self._size))
        self.gamma = gamma
        self.zeta = zeta
        self.gamma_zeta_ratio = math.log(-gamma / zeta)
        self.sigmoid = nn.Sigmoid()
    
    def _hard_sigmoid(self, x):
        return torch.min(torch.max(x, torch.zeros_like(x)), torch.ones_like(x))

    def _get_mask(self):
        if self.training:
            self.uniform.uniform_()
            u = Variable(self.uniform)
            s = self.sigmoid((torch.log(u) - torch.log(1 - u) + self.loc) / self.temp)
            s = s * (self.zeta - self.gamma) + self.gamma
            penalty = self.sigmoid(self.loc - self.temp * self.gamma_zeta_ratio).sum()
        else:
            s = self.sigmoid(self.loc) * (self.zeta - self.gamma) + self.gamma
            penalty = 0
        return self._hard_sigmoid(s), penalty


class L0Linear(_L0Norm):
    def __init__(self, in_features, out_features, bias=True, **kwargs):
        super(L0Linear, self).__init__(nn.Linear(in_features, out_features, bias=bias), **kwargs)

    def forward(self, input):
        mask, penalty = self._get_mask()
        return F.linear(input, self._origin.weight * mask, self._origin.bias), penalty

class ConstrainedLinear(torch.nn.Module):
    def __init__ (self, input_size, output_size, hard_init=None, bias=True): 
        super().__init__()
        if hard_init is None:
            print('[INFO] Random decoder initialization.')
            self.W = nn.Parameter(torch.zeros(input_size, output_size)) 
            self.W = nn.init.kaiming_normal_(self.W)
        else:
            print('[INFO] Hardcoded decoder initialization.')
            try:
                assert hard_init.size()[0] == input_size
                assert hard_init.size()[1] == output_size
            except AssertionError as ae:
                ae.args += (f'Decoder initialization tensor does not have the required input size.\n Received: {tuple(hard_init.size())}\n Needed: ({input_size}, {output_size})\n',)
                raise ae
            except Exception as e:
                raise e
            self.W = nn.Parameter(hard_init)
        self.bias = bias
        if self.bias:
            self.b = nn.Parameter(torch.ones(output_size)) 

    def forward(self, x):
        if self.bias:
            return torch.addmm(self.b, x, torch.sigmoid(self.W))
        return torch.mm(x, torch.sigmoid(self.W)) 


class AdmixtureAE(torch.nn.Module):
    def __init__(self, k, num_features, beta_l0=2/3, gamma_l0=-0.1, zeta_l0=1.1, lambda_l0=0.1, P_init=None, deep_encoder=False):
        super().__init__()
        self.k = k
        self.num_features = num_features
        self.beta_l0, self.gamma_l0, self.zeta_l0 = beta_l0, gamma_l0, zeta_l0
        self.lambda_l0 = lambda_l0
        self.deep_encoder = deep_encoder
        if lambda_l0 > 0:
            self.encoder = L0Linear(self.num_features, self.k, bias=True, beta=self.beta_l0, gamma=self.gamma_l0, zeta=self.zeta_l0)
        else:
            if not self.deep_encoder:
                self.encoder = nn.Linear(self.num_features, self.k, bias=True)
            else:
                self.encoder = nn.Sequential(
                        nn.Linear(self.num_features, 512, bias=True),
                        nn.Linear(512, 128, bias=True),
                        nn.Linear(128, 32, bias=True),
                        nn.Linear(32, self.k, bias=True)
                )
        self.decoder = ConstrainedLinear(self.k, num_features, hard_init=P_init, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        if self.lambda_l0 > 0:
            enc, l0_pen = self.encoder(X)
        else:
            enc = self.encoder(X)
            l0_pen = 0
        hid_state = self.softmax(enc)
        reconstruction = self.decoder(hid_state)
        return reconstruction, hid_state, l0_pen/X.shape[0]
        
    def train(self, trX, optimizer, loss_f, num_epochs, device, batch_size=0, valX=None, display_logs=True, loss_weights=None, save_every=10, save_path='../outputs/model.pt'):
        for ep in range(num_epochs):
            tr_loss, val_loss = self._run_epoch(trX, optimizer, loss_f, batch_size, valX, device, loss_weights)
            if display_logs and ep % 50 == 0:
                print(f'EPOCH {ep+1}: mean training loss: {tr_loss}')
                if val_loss is not None:
                    print(f'EPOCH {ep+1}: mean validation loss: {val_loss}')
            if save_every*ep > 0 and ep % save_every == 0:
                torch.save(self.state_dict(), save_path)
        return tr_loss, val_loss

    def _batch_generator(self, X, batch_size=0):
        if batch_size < 1:
            yield torch.tensor(X, dtype=torch.float32)
        else:
            for i in range(0, X.shape[0], batch_size):
                yield torch.tensor(X[i:i+batch_size], dtype=torch.float32)

    def _validate(self, valX, loss_f, batch_size, device):
        acum_val_loss = 0
        with torch.no_grad():
            for X in self._batch_generator(valX, batch_size):
                rec, _, _ = self.forward(X.to(device))
                acum_val_loss += loss_f(rec, X).cpu().item()
        return acum_val_loss

        
    def _run_step(self, X, optimizer, loss_f, loss_weights=None):
        optimizer.zero_grad()
        rec, _, l0_pen = self.forward(X)
        if loss_weights is not None:
            loss = loss_f(rec, X, loss_weights)
        else:
            loss = loss_f(rec, X)
        if self.lambda_l0 > 0:
            loss += self.lambda_l0*l0_pen
        loss.backward()
        optimizer.step()
        return loss

    def _run_epoch(self, trX, optimizer, loss_f, batch_size, valX, device, loss_weights=None):
        tr_loss, val_loss = 0, None
        for X in self._batch_generator(trX, batch_size):
            step_loss = self._run_step(X.to(device), optimizer, loss_f, loss_weights)
            tr_loss += step_loss.cpu().item()
        if valX is not None:
            val_loss = self._validate(valX, loss_f, device)
            return tr_loss / trX.shape[0], val_loss / valX.shape[0]
        return tr_loss / trX.shape[0], None
