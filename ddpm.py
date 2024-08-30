import torch



def myfunc():



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

# Forward Diffusion
#for _ in range(T):
#    pass
