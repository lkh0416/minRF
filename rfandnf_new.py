# implementation of Rectified Flow for simple minded people like me.
import argparse

import torch
import torch.nn.functional as F

class RF:
    def __init__(self, model, ln=True):
        self.model = model
        self.ln = ln

    def forward(self, x, cond):
        b = x.size(0)
        if self.ln:
            nt = torch.randn((b,)).to(x.device)
            t = torch.sigmoid(nt)
        else:
            t = torch.rand((b,)).to(x.device)
        texp = t.view([b, *([1] * len(x.shape[1:]))])
        z1 = torch.randn_like(x)
        zt = (1 - texp) * x + texp * z1
        vtheta = self.model(zt, t, cond)
        batchwise_mse = ((z1 - x - vtheta) ** 2).mean(dim=list(range(1, len(x.shape))))
        tlist = batchwise_mse.detach().cpu().reshape(-1).tolist()
        ttloss = [(tv, tloss) for tv, tloss in zip(t, tlist)]
        return batchwise_mse.mean(), ttloss

    @torch.no_grad()
    def sample(self, z, cond, null_cond=None, sample_steps=50, cfg=2.0):
        b = z.size(0)
        dt = 1.0 / sample_steps
        dt = torch.tensor([dt] * b).to(z.device).view([b, *([1] * len(z.shape[1:]))])
        images = [z]
        for i in range(sample_steps, 0, -1):
            t = i / sample_steps
            t = torch.tensor([t] * b).to(z.device)

            vc = self.model(z, t, cond)
            if null_cond is not None:
                vu = self.model(z, t, null_cond)
                vc = vu + cfg * (vc - vu)

            z = z - dt * vc
            images.append(z)
        return images

    def reverse_sample(self, y1, cond, null_cond=None, sample_steps=2, cfg=2.0):
        b = y1.size(0)
        dt = 1.0 / sample_steps
        dt = torch.tensor([dt] * b).to(y1.device).view([b, *([1] * len(y1.shape[1:]))])

        z = y1
        for i in range(1, sample_steps + 1):
            t = i / sample_steps
            t = torch.tensor([t] * b).to(y1.device)

            vc = self.model(z, t, cond)
            if null_cond is not None:
                vu = self.model(z, t, null_cond)
                vc = vu + cfg * (vc - vu)

            z = z + dt * vc

        return z

def elbo_loss(rf, x, c, sample_steps):
    """
    Compute the ELBO for a given input x and conditional labels c.
    Args:
        rf: Reverse diffusion model (RF instance).
        x: Input data (batch of images).
        c: Conditional information (class labels).
        sample_steps: Number of reverse diffusion steps.
    
    Returns:
        elbo: ELBO approximation (loss) for the batch.
    """
    # Reverse sample to obtain z from x
    z = rf.reverse_sample(x, c, sample_steps=sample_steps)
    
    # We use the model to predict the reverse process (denoising step) to estimate the loss
    batch_size = x.size(0)
    elbo_loss_total = 0.0
    
    for i in range(1, sample_steps + 1):
        t = torch.tensor([i / sample_steps] * batch_size).to(x.device)
        texp = t.view([batch_size, *([1] * len(x.shape[1:]))])
        
        # Add noise for current time step
        z_noisy = (1 - texp) * x + texp * torch.randn_like(x)
        
        # Get model's prediction for the reverse step
        vtheta = rf.model(z_noisy, t, c)
        
        # The target is the original sample minus the noise (similar to MSE)
        target = torch.randn_like(x)
        
        # Compute the MSE between the model prediction and the true noise
        elbo_loss_step = F.mse_loss(vtheta, target)
        elbo_loss_total += elbo_loss_step
    
    # Return the average ELBO across all reverse diffusion steps
    return elbo_loss_total / sample_steps

def compute_elbo_for_sample(rf, z, c):
    """
    Compute the ELBO for a given sample z.
    Args:
        rf: Reverse diffusion model (RF instance).
        z: Sampled latent representation.
        c: Conditional information (class labels).
    
    Returns:
        elbo: ELBO approximation (loss) for the sample.
    """
    # We use the model to predict the reverse process (denoising step) to estimate the loss
    batch_size = z.size(0)
    elbo_loss_total = 0.0
    
    for i in range(1, sample_steps + 1):
        t = torch.tensor([i / sample_steps] * batch_size).to(z.device)
        texp = t.view([batch_size, *([1] * len(z.shape[1:]))])
        
        # Add noise for current time step
        z_noisy = (1 - texp) * z + texp * torch.randn_like(z)
        
        # Get model's prediction for the reverse step
        vtheta = rf.model(z_noisy, t, c)
        
        # The target is the original sample minus the noise (similar to MSE)
        target = torch.randn_like(z)
        
        # Compute the MSE between the model prediction and the true noise
        elbo_loss_step = F.mse_loss(vtheta, target)
        elbo_loss_total += elbo_loss_step
    
    # Return the average ELBO across all reverse diffusion steps
    return elbo_loss_total / sample_steps

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

if __name__ == "__main__":
    # train class conditional RF on mnist.
    import numpy as np
    import torch.optim as optim
    from PIL import Image
    import copy
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms
    from torchvision.utils import make_grid
    from tqdm import tqdm

    import wandb
    from dit import DiT_Llama
    from itertools import zip_longest

    parser = argparse.ArgumentParser(description="use cifar?")
    parser.add_argument("--cifar", action="store_true")
    parser.add_argument("--th", default=0.15, type=float)
    args = parser.parse_args()
    CIFAR = args.cifar

    if CIFAR:
        dataset_name = "cifar"
        fdatasets = datasets.CIFAR10
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        channels = 3
        model = DiT_Llama(
            channels, 32, dim=256, n_layers=10, n_heads=8, num_classes=10
        ).cuda()
    else:
        dataset_name = "mnist"
        fdatasets = datasets.MNIST
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Pad(2),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        channels = 1
        model = DiT_Llama(
            channels, 32, dim=64, n_layers=6, n_heads=4, num_classes=10
        ).cuda()

    # model parameters
    forget_threshold = 4.0  # δ from the paper
    error_bound = 0.15 * forget_threshold  # ε = 0.15 * δ
    alpha = 0.6  # Combination factor for losses, as indicated in the supplementary material
    gamma = 0.6  # Factor for KL-divergence
    learning_rate = 1e-4  # η
    sample_steps = 2 # 5
    batch_size = 24 # 8

    # Load CIFAR-10 dataset to check out the labels
    # Class : ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    full_dataset = fdatasets(root="./data", train=True, download=True, transform=transform)

    # Define the specific class you want to separate (e.g., 'airplane' class, label 0)
    forget_class = 0  # Label for 'airplane'
    class_names = full_dataset.classes  # ['airplane', 'automobile', ..., 'truck']

    # Separate the dataset into two subsets:
    forget_class_indices = [i for i, (_, label) in enumerate(full_dataset) if label == forget_class]
    remember_class_indices = [i for i, (_, label) in enumerate(full_dataset) if label != forget_class]
    
    # Create Subsets
    forget_class_dataset = Subset(full_dataset, forget_class_indices)
    remember_class_dataset = Subset(full_dataset, remember_class_indices)

    # Check the sizes of the subsets
    print(f"Number of samples in forget class ({class_names[forget_class]}): {len(forget_class_dataset)}")
    print(f"Number of samples in remember classes: {len(remember_class_dataset)}")

    # Create DataLoaders for each subset
    forget_class_loader = DataLoader(forget_class_dataset, batch_size=batch_size, shuffle=True)
    remember_class_loader = DataLoader(remember_class_dataset, batch_size=batch_size, shuffle=True)

    # load pretrained model for cifar 'model.pt'
    model.load_state_dict(torch.load('model.pt'))

    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {model_size}, {model_size / 1e6}M")

    # make copy of model to tame
    model_taming = copy.deepcopy(model)

    # fixing model parameters of base model
    for param in model.parameters():
        param.requires_grad = False

    # We have to tame 'model_taming' based on 'model'.
    rf = RF(model)
    rf_taming = RF(model_taming)

    wandb.init(project=f"rf_taming_new_{dataset_name}")

    optimizer = optim.SGD(model_taming.parameters(), lr=learning_rate)

    for epoch in range(100):
        lossbin = {i: 0 for i in range(10)}
        losscnt = {i: 1e-6 for i in range(10)}
        for (forget_batch, remember_batch) in tqdm(zip_longest(forget_class_loader, remember_class_loader)):
            if forget_batch == None or remember_batch == None:
                # both batch must exist to perform forward pass, so break if one of them is None
                break
            
            if forget_batch is not None:
                x_forget, c_forget = forget_batch
                x_forget, c_forget = x_forget.cuda(), c_forget.cuda()

            if remember_batch is not None:
                x_remember, c_remember = remember_batch
                x_remember, c_remember = x_remember.cuda(), c_remember.cuda()

            optimizer.zero_grad()

            # print(z.mean().item(), z.std().item())

            # Calculate forget loss
            elbo_likelihood = elbo_loss(rf_taming, x_forget, c_forget, sample_steps)

            with torch.no_grad():
                elbo_loss_batch = []
                for i in range(batch_size):
                    # Reverse sample for each individual sample
                    z = rf_taming.reverse_sample(x_remember[i:i+1], c_remember[i:i+1], sample_steps=sample_steps)
                    elbo_loss_sample = compute_elbo_for_sample(rf_taming, z, c_remember[i:i+1])  # Assuming this is defined
                    elbo_loss_batch.append(elbo_loss_sample)

                elbo_mean_remember = torch.mean(torch.stack(elbo_loss_batch))  # Mean over batch
                elbo_std_remember = torch.std(torch.stack(elbo_loss_batch))    # Standard deviation over batch

                # print(elbo_mean, elbo_std)

            for i in range(batch_size):
                # Reverse sample for each individual sample
                z = rf_taming.reverse_sample(x_remember[i:i+1], c_remember[i:i+1], sample_steps=sample_steps)
                elbo_loss_sample = compute_elbo_for_sample(rf_taming, z, c_remember[i:i+1])  # Assuming this is defined
                elbo_loss_batch.append(elbo_loss_sample)

            elbo_mean = torch.mean(torch.stack(elbo_loss_batch))  # Mean over batch
            elbo_std = torch.std(torch.stack(elbo_loss_batch))    # Standard deviation over batch

            elbo = (elbo_loss(rf_taming, x_remember, c_remember, sample_steps) - elbo_mean_remember - forget_threshold * elbo_std_remember) / elbo_std_remember
            loss_forget = torch.sigmoid((elbo_std**2) * (elbo**2)) / batch_size

            # calculate remember loss
            elbo_remember = elbo_loss(rf_taming, x_remember, c_remember, sample_steps)
            kl_forget_1 = kl_loss(rf, rf_taming, x_remember, c_remember, sample_steps)
            kl_forget_2 = kl_loss(rf_taming, rf, x_remember, c_remember, sample_steps)
            loss_remember = (1 - gamma) * elbo_remember + gamma * (kl_forget_1 + kl_forget_2)

            # caculate tota loss
            loss  = alpha * loss_forget + (1 - alpha) * loss_remember

            loss.backward()
            optimizer.step()

            # wandb.log({"loss": loss.item()})
            wandb.log({"loss": loss.item(), "loss_forget": loss_forget.item(), "loss_remember": loss_remember.item()})

    #         x, c = x.cuda(), c.cuda()
    #         optimizer.zero_grad()
    #         loss, blsct = rf.forward(x, c)
    #         loss.backward()
    #         optimizer.step()

    #         wandb.log({"loss": loss.item()})

    #         # count based on t
    #         for t, l in blsct:
    #             lossbin[int(t * 10)] += l
    #             losscnt[int(t * 10)] += 1

    #     # log
    #     for i in range(10):
    #         print(f"Epoch: {epoch}, {i} range loss: {lossbin[i] / losscnt[i]}")

    #     wandb.log({f"lossbin_{i}": lossbin[i] / losscnt[i] for i in range(10)})

    #     # save model
    #     torch.save(rf.model.state_dict(), 'model.pt')

        rf_taming.model.eval()
        with torch.no_grad():
            cond = torch.arange(0, 16).cuda() % 10
            uncond = torch.ones_like(cond) * 10

            init_noise = torch.randn(16, channels, 32, 32).cuda()
            images = rf_taming.sample(init_noise, cond, uncond)
            # image sequences to gif
            gif = []
            for image in images:
                # unnormalize
                image = image * 0.5 + 0.5
                image = image.clamp(0, 1)
                x_as_image = make_grid(image.float(), nrow=4)
                img = x_as_image.permute(1, 2, 0).cpu().numpy()
                img = (img * 255).astype(np.uint8)
                gif.append(Image.fromarray(img))

            gif[0].save(
                f"contents/sample_{epoch}.gif",
                save_all=True,
                append_images=gif[1:],
                duration=100,
                loop=0,
            )

            last_img = gif[-1]
            last_img.save(f"contents/sample_{epoch}_last.png")

        rf_taming.model.train()