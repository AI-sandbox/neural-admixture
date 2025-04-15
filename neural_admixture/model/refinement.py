from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from collections import deque

import multiprocessing
import torch.nn as nn
import torch.optim as optim
import torch
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

class AdmixtureDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return idx, self.data[idx].masked_fill(self.data[idx] == 9, 0).float() / 2

class AdmixtureOptimizer(nn.Module):
    def __init__(self, init_P, init_Q):
        super().__init__()
        self.P = nn.Parameter(init_P)
        self.Q = nn.Parameter(init_Q)
    
    def forward(self, indices):
        Q_batch = self.Q[indices]
        return torch.clamp_(Q_batch @ self.P, 0, 1)

def refine_Q_P(P, Q, data, device, num_cpus, num_epochs=5, patience=2):

    multiprocessing.set_start_method('fork', force=True)
    torch.set_float32_matmul_precision('medium')
    torch.set_flush_denormal(True)
    
    dataset = AdmixtureDataset(data)
    batch_size = min(10240, data.shape[0])
    log.info(f"    Using {batch_size} batch size.")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=max(2, min(num_cpus - 1, int(num_cpus * 0.065))), persistent_workers=True) #prefetch_factor=4
    
    model = AdmixtureOptimizer(P, Q)
    #model = torch.compile(model)
    model = model.to(device)
    loss_f = nn.BCELoss()
    
    best_logl = float('inf')
    best_P = model.P.clone().detach()
    best_Q = model.Q.clone().detach()
    recent_losses = deque(maxlen=patience)
    
    accum_steps = len(dataloader)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, fused=True)

    for epoch in tqdm(range(num_epochs), desc="Epochs", file=sys.stderr):
        loss_acc = 0.0
                
        optimizer.zero_grad(set_to_none=True)
        
        for i, (indices, X_batch) in enumerate(dataloader):
            indices = indices.to(device, non_blocking=True)
            X_batch = X_batch.to(device, non_blocking=True)

            # Forward
            #with torch.amp.autocast(device_type=model.P.device.type):
            R = model(indices)
            loss = loss_f(R, X_batch)
            loss_acc += loss.item()

            # Backward with scaled loss
            loss = loss / accum_steps
            loss.backward()

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            model.P.clamp_(0.0, 1.0)
            model.Q.clamp_(0.0, 1.0)
        
        log.info(f"    Epoch {epoch+1}, Loss: {loss_acc:.7f}")
        
        if loss_acc < best_logl:
            best_logl = loss_acc
            best_P = model.P.clone().detach()
            best_Q = model.Q.clone().detach()

        recent_losses.append(loss_acc)
        
        if len(recent_losses) == patience and not any(loss < recent_losses[0] for loss in list(recent_losses)[1:]):
            log.info("")
            log.info("    Early stopping triggered")
            log.info("")
            break

    return best_P.cpu(), best_Q.cpu()

def loglikelihood(Q: torch.Tensor, P: torch.Tensor, data: torch.Tensor, batch_size: int = 512, eps: float = 1e-7) -> float:
    """Compute deviance for a single K using PyTorch tensors

    Args:
        Q (torch.Tensor): Matrix Q of shape (N, D).
        P (torch.Tensor): Matrix P of shape (D, M).
        data (torch.Tensor): Original data of shape (N, M).
        batch_size (int, optional): Size of each batch. Defaults to 64.
        eps (float, optional): Epsilon term to avoid numerical errors. Defaults to 1e-7.

    Returns:
        float: Computed log-likelihood value.
    """
    result = 0.0
    N = Q.shape[0]

    for init in range(0, N, batch_size):
        end = min(init + batch_size, N)
        Q_batch = Q[init:end]
        data_batch = data[init:end]

        mask = data_batch != 9
        rec_batch = torch.clamp(torch.matmul(Q_batch, P), eps, 1 - eps)
        data_batch = torch.clamp(data_batch, eps, 2 - eps)

        loglikelihood_batch = data_batch * torch.log(rec_batch) + (2 - data_batch) * torch.log1p(-rec_batch)

        result += torch.sum(loglikelihood_batch[mask]).item()

    return result