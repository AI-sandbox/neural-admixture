import torch
from torch import Tensor
from torch.autograd import Variable

class CustomWBCE():
    def __call__(self, output:Tensor, target:Tensor, weights:Tensor, **kwargs)->Tensor:        
        if output.min() <= 0 or output.max() >= 1:
            output = torch.clamp(torch.sigmoid(output),min=1e-8,max=1 - 1e-8)
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)
        loss = torch.neg(torch.mean(torch.mul(loss, weights)))
        return loss