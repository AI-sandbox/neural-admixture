import numpy as np
import torch
from torch import Tensor
from torch.autograd import Variable

class WeightedBCE:
    def __call__(self, output:Tensor, target:Tensor, weights:Tensor, **kwargs)->Tensor:
        if output.min() <= 0 or output.max() >= 1:
            output = torch.clamp(torch.sigmoid(output),min=1e-8,max=1 - 1e-8)
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)
        loss = torch.neg(torch.mean(torch.mul(loss, weights)))
        return loss

class WeightedMSE:
    def __init__(self):
        raise NotImplementedError
    def __call__(self, output:Tensor, target:Tensor, weights:Tensor, **kwargs)->Tensor:
        pass # TO-DO

class MaskedBCE:
    def __init__(self, device, mask_idxs=None, mask_frac=None):
        try:
            assert mask_idxs is not None or mask_frac is not None
        except AssertionError as ae:
            ae.args += ('Either mask indices or a mask fraction must be provided',)
            raise ae
        except Exception as e:
            raise e
        self.mask_frac = mask_frac
        self.device = device
        self.mask_sel = 1-mask_idxs if mask_idxs is not None else None

    def __call__(self, output:Tensor, target:Tensor, **kwargs)->Tensor:
        if output.min() <= 0 or output.max() >= 1:
            output = torch.clamp(torch.sigmoid(output),min=1e-8,max=1 - 1e-8)
        if self.mask_sel is not None:
            mask_sel = torch.tensor(np.tile(self.mask_sel, (output.shape[0], 1)), device=self.device, dtype=torch.float)
        else:
            mask_sel = torch.tensor(np.tile(np.random.binomial(1, self.mask_frac, output.shape[1]), (output.shape[0],1)), device=self.device, dtype=torch.float)
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)
        return torch.neg(torch.mean(torch.mul(loss, mask_sel)))