from typing import Tuple, Union
import torch
import torch.nn as nn
from unet import Unet


# Define number of steps to add noise to input
T = 1000

# Noise schedule at each time steps (beta_{i})
beta_start = 1e-3
beta_end = 2e-2
betas = torch.linspace(beta_start, beta_end, steps=T)

# Calculate alpha_{i}
alphas = 1 - betas

# Calculate alpha_bar_{i}
alphas_bar = []
prev_prod = 1.
for step in range(T):
    _alpha_bar = alphas[step] * prev_prod
    alphas_bar.append(_alpha_bar)
    prev_prod = _alpha_bar


x_0 = 0. # need to define this quantity

noised_x = []
# Forward Diffusion
for i in range(T):
    # sampling x_t at time step t
    if not isinstance(x_0, torch.Tensor):
        raise ValueError(f'Input {x_0} should be a torch tensor.')
    x_i = torch.sqrt(alphas_bar[i]) * x_0 + torch.sqrt(1 - alphas_bar[i]) * torch.normal(mean=0., std=1., x_0.shape)
    noised_x.append(x_i)

# Reverse process
def sample(
        unet: nn.Module,
        x_T: torch.Tensor,
        steps: int,
        tensor_shape: Union[Tuple[int, int], torch.Tensor],
        seed: int = 42
    ):
    x = []
    x_curr = x_T
    for t in reversed(range(steps)):
        eps_t = torch.normal(mean=0., std=1., size=tensor_shape)
        x_prev = (1 / torch.sqrt(alphas[t])) * (x_curr - ((1 - alphas[t]) / (torch.sqrt(1 - alphas_bar[t])))) * unet(x_curr, t)) + torch.sqrt(betas[t]) * eps_t
        x.insert(0, x_prev)
        x_curr = x_prev
    return x[0]

# Training Loss
while




















