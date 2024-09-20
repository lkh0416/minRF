import argparse
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
from losses_minrf import elbo_loss, kl_divergence, compute_elbo_for_sample, compute_nll

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

def show_images(images, title):
    images = images * 0.5 + 0.5
    images = images.clamp(0, 1)
    # img = images.permute(1, 2, 0).cpu().numpy()
    img = images.cpu().numpy()
    img = (img * 255).astype(np.uint8)
    grid_img = make_grid(images, nrow=10)
    plt.figure(figsize=(10, 10))
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.title(title)
    plt.axis('off')
    plt.savefig(title)

if __name__ == "__main__":
    import numpy as np
    import torch.optim as optim
    from PIL import Image
    import copy
    from torchvision.utils import make_grid
    from tqdm import tqdm

    import wandb
    from dit import DiT_Llama
    from itertools import zip_longest

    parser = argparse.ArgumentParser(description="use cifar?")
    parser.add_argument("--cifar", action="store_true")
    parser.add_argument("--forget_percentage", default=0.5, type=float, help="Percentage of forget class to use for forgetting")
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
    forget_threshold = 1.0
    alpha = 0.6  
    gamma = 0.6  
    learning_rate = 1e-1
    sample_steps = 1
    batch_size = 64

    # Load CIFAR-10 or MNIST dataset
    full_dataset = fdatasets(root="./data", train=True, download=True, transform=transform)

    # Define the specific class you want to separate (e.g., 'airplane' class, label 0)
    forget_class = 0  # For CIFAR-10, 0 is 'airplane'. You can change this as needed.
    class_names = full_dataset.classes

    # Separate the dataset into two subsets:
    certain_class_indices = [i for i, (_, label) in enumerate(full_dataset) if label == forget_class]

    # Define the portion of the forget class you want to forget (e.g., 50%)
    forget_percentage = args.forget_percentage
    num_forget_samples = int(len(certain_class_indices) * forget_percentage)

    # Create the subset of certain_class_indices for forgetting
    forget_subset_indices = certain_class_indices[:num_forget_samples]

    # Create the subset of certain_class_indices for remembering
    remember_subset_indices = certain_class_indices[num_forget_samples:]

    # Create Subsets
    forget_class_dataset = Subset(full_dataset, forget_subset_indices)
    remember_class_dataset = Subset(full_dataset, remember_subset_indices)

    # Check the sizes of the subsets
    print(f"Number of samples in forget class ({class_names[forget_class]}): {len(forget_class_dataset)}")
    print(f"Number of samples in remember classes: {len(remember_class_dataset)}")

    # Create DataLoaders for each subset
    forget_class_loader = DataLoader(forget_class_dataset, batch_size=batch_size, shuffle=True)
    remember_class_loader = DataLoader(remember_class_dataset, batch_size=batch_size, shuffle=True)

    #### Checking data
    # Visualize forget_class_loader
    forget_images, _ = next(iter(forget_class_loader))
    show_images(forget_images, f"Forget Class ({class_names[forget_class]}).png")

    # # Visualize 100 samples of remember_class_loader
    # remember_images, _ = next(iter(remember_class_loader))
    # show_images(remember_images[:100], "Remember Classes (100 samples).png")

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

            # Calculate forget loss
            elbo_likelihood = elbo_loss(rf_taming, x_forget, c_forget, sample_steps)

            with torch.no_grad():
                elbo_loss_batch = []
                for i in range(batch_size):
                    # Reverse sample for each individual sample
                    z = rf_taming.reverse_sample(x_remember[i:i+1], c_remember[i:i+1], sample_steps=sample_steps)
                    elbo_loss_sample = compute_elbo_for_sample(rf_taming, z, c_remember[i:i+1], sample_steps)  # Assuming this is defined
                    elbo_loss_batch.append(elbo_loss_sample)

                elbo_mean_remember = torch.mean(torch.stack(elbo_loss_batch))  # Mean over batch
                elbo_std_remember = torch.std(torch.stack(elbo_loss_batch))    # Standard deviation over batch

                print(f"elbo_mean_remember: {elbo_mean_remember}, elbo_std_remember: {elbo_std_remember}")


            ## calculating the actual forgetting loss
            loss_forget = 0.0
            for i in range(len(x_forget)):
                # Reverse sample for each individual sample
                z = rf_taming.reverse_sample(x_forget[i:i+1], c_forget[i:i+1], sample_steps=sample_steps)
                print(f"z mean : {z.mean()}, z std : {z.std()}")
                elbo_loss_sample = compute_elbo_for_sample(rf_taming, z, c_forget[i:i+1], sample_steps=sample_steps)  # Assuming this is defined
                # elbo = (elbo_loss_sample - elbo_mean_remember - forget_threshold * elbo_std_remember) / elbo_std_remember
                # elbo = elbo_loss_sample
                elbo = (elbo_loss_sample - elbo_mean_remember - forget_threshold * elbo_std_remember)
                single_loss = torch.sigmoid((elbo_std_remember**2) * (elbo**2))
                print(f"elbo: {elbo.item()}, loss: {single_loss.item()}")
                loss_forget += single_loss
            

            # print(elbo.item())
            # print(((elbo_std_remember**2) * (elbo**2)).item())

            # loss_forget = torch.sigmoid((elbo_std_remember**2) * (elbo**2)) / batch_size
            loss_forget = loss_forget / len(x_forget)
            
            # # Calculate forget loss
            # nll_likelihood = compute_nll(rf_taming, x_forget, c_forget, sample_steps)

            # with torch.no_grad():
            #     nll_loss_batch = []
            #     for i in range(batch_size):
            #         # Reverse sample for each individual sample
            #         z = rf_taming.reverse_sample(x_remember[i:i+1], c_remember[i:i+1], sample_steps=sample_steps)
            #         nll_loss_sample = compute_nll(rf_taming, z, c_remember[i:i+1], sample_steps)
            #         nll_loss_batch.append(nll_loss_sample)

            #     nll_mean_remember = torch.mean(torch.stack(nll_loss_batch))  # Mean over batch
            #     nll_std_remember = torch.std(torch.stack(nll_loss_batch))    # Standard deviation over batch

            #     print(nll_mean_remember, nll_std_remember)


            # ## calculating the actual forgetting loss
            # loss_forget = 0.0
            # for i in range(len(x_forget)):
            #     # Reverse sample for each individual sample
            #     z = rf_taming.reverse_sample(x_forget[i:i+1], c_forget[i:i+1], sample_steps=sample_steps)
            #     nll_loss_sample = compute_nll(rf_taming, z, c_forget[i:i+1], sample_steps)
            #     print(nll_loss_sample.item(), end=" ")
            #     nll = (nll_loss_sample - nll_mean_remember - forget_threshold * nll_std_remember) / nll_std_remember
            #     print(nll.item(), end=" ")
            #     single_loss = torch.sigmoid((nll_std_remember**2) * (nll**2))
            #     print(single_loss)
            #     loss_forget += single_loss
            

            # print(nll.item())
            # print(((nll_std_remember**2) * (nll**2)).item())

            # loss_forget = torch.sigmoid((nll_std_remember**2) * (nll**2)) / batch_size
            loss_forget = loss_forget / len(x_forget)

            # # calculate remember loss
            # elbo_remember = elbo_loss(rf_taming, x_remember, c_remember, sample_steps)
            # kl_forget_1 = kl_loss(rf, rf_taming, x_remember, c_remember, sample_steps)
            # kl_forget_2 = kl_loss(rf_taming, rf, x_remember, c_remember, sample_steps)
            # loss_remember = (1 - gamma) * elbo_remember + gamma * (kl_forget_1 + kl_forget_2)

            # caculate total loss
            # loss  = alpha * loss_forget + (1 - alpha) * loss_remember
            loss = loss_forget
            # loss = loss_remember

            loss.backward()
            optimizer.step()

            # wandb.log({"loss": loss.item()})
            # wandb.log({"loss": loss.item(), "loss_remember": loss_remember.item()})
            wandb.log({"loss": loss.item(), "loss_forget": loss_forget.item(), "elbo value": torch.abs(elbo).item()})
            # wandb.log({"loss": loss.item(), "loss_forget": loss_forget.item(), "loss_remember": loss_remember.item()})

        rf_taming.model.eval()
        with torch.no_grad():
            cond = torch.tensor([forget_class] * 16).cuda()
            uncond = torch.ones_like(cond) * 10

            init_noise = torch.randn(16, channels, 32, 32).cuda()
            # print(f"init_noise mean : {init_noise.mean()}, init_noise std : {init_noise.std()}")
            images = rf_taming.sample(init_noise, cond, uncond)
            # print(f"images mean : {images[-1].mean()}, images std : {images[-1].std()}")
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