import torch
import torch.nn.functional as F

def elbo_loss(rf, x, c, sample_steps):
    z = rf.reverse_sample(x, c, sample_steps=sample_steps)
    batch_size = x.size(0)
    elbo_loss_total = 0.0
    
    for i in range(1, sample_steps + 1):
        t = torch.tensor([i / sample_steps] * batch_size).to(x.device)
        texp = t.view([batch_size, *([1] * len(x.shape[1:]))])
        z_noisy = (1 - texp) * x + texp * torch.randn_like(x)
        vtheta = rf.model(z_noisy, t, c)
        target = torch.randn_like(x)
        elbo_loss_step = F.mse_loss(vtheta, target)
        elbo_loss_total += elbo_loss_step
    
    return elbo_loss_total / sample_steps

# function to calculate elbo for a random variable?
def compute_elbo_for_sample(rf, z, c, sample_steps):
    batch_size = z.size(0)
    elbo_loss_total = 0.0
    
    for i in range(1, sample_steps + 1):
        t = torch.tensor([i / sample_steps] * batch_size).to(z.device)
        texp = t.view([batch_size, *([1] * len(z.shape[1:]))])
        z_noisy = (1 - texp) * z + texp * torch.randn_like(z)
        vtheta = rf.model(z_noisy, t, c)
        target = torch.randn_like(z)
        elbo_loss_step = F.mse_loss(vtheta, target)
        elbo_loss_total += elbo_loss_step
    
    return elbo_loss_total / sample_steps

def negative_log_likelihood(model, x_batch, c_batch, sample_steps):
    # Reverse sample to get the reconstructed batch
    x_reconstructed = model.reverse_sample(x_batch, c_batch, sample_steps=sample_steps)
    
    # Flatten the input and reconstructed batches
    x_batch_flat = x_batch.view(x_batch.shape[0], -1)
    x_reconstructed_flat = x_reconstructed.view(x_batch.shape[0], -1)
    
    # Compute the negative log-likelihood using Mean Squared Error
    nll = F.mse_loss(x_reconstructed_flat, x_batch_flat, reduction='mean')
    
    return nll

def compute_nll(model, x_batch, c_batch, sample_steps):
    x0 = model.reverse_sample(x_batch, c_batch, sample_steps=sample_steps)
    # x1 = x_batch

    # (256, 1, 32, 32) -> (256, 32*32)로 reshape
    x0_flat = x0.view(x_batch.shape[0], -1)
    # x1_flat = x1.view(x_batch.shape[0], -1)

    # x0와 x1의 차이 계산
    # diff = x0_flat - x1_flat
    diff = x0_flat

    # 차이에 대한 L2 norm 계산 (dim=1을 기준으로 계산)
    l2_norm_diff = torch.norm(diff, p=2, dim=1)

    return l2_norm_diff

def kl_divergence(mu_q, sigma_q, mu_p, sigma_p):
    """Compute KL divergence between two Gaussian distributions"""
    kl = torch.log(sigma_p / sigma_q) + (sigma_q**2 + (mu_q - mu_p)**2) / (2 * sigma_p**2) - 0.5
    # print(kl)
    # print(kl.shape)
    return kl.sum(dim=0).mean()

def kl_loss(rf, rf_taming, x_set, c_set, sample_steps):
    # Get the reverse sample for both models
    z_base = rf.reverse_sample(x_set, c_set, sample_steps=sample_steps)
    z_taming = rf_taming.reverse_sample(x_set, c_set, sample_steps=sample_steps)

    # Compute mean and variance (assuming a Gaussian distribution) for both
    mean_taming = z_taming.mean(dim=[1, 2, 3])  # Compute the mean across spatial dimensions
    std_taming = z_taming.std(dim=[1, 2, 3])    # Compute the standard deviation

    mean_base = z_base.mean(dim=[1, 2, 3])
    std_base = z_base.std(dim=[1, 2, 3])

    # Compute the KL divergence between the "base" and "taming" distributions
    kl_div = kl_divergence(mean_base, std_base, mean_taming, std_taming)

    return kl_div
