# implementation of Rectified Flow for simple minded people like me.
import argparse
# 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
import torch


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
    def sample(self, z, cond, null_cond=None, sample_steps=2, cfg=2.0):
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

    # @torch.no_grad()
    def reverse_sample(self, y1, cond, null_cond=None, sample_steps=2, cfg=2.0):
        """
        이미지 Y1에서 noise Y0으로 역과정 추정

        y1: 초기 이미지 (torch.Tensor)
        cond: 조건 (torch.Tensor)
        null_cond: null 조건 (torch.Tensor), 필요 시 사용
        sample_steps: 역방향 샘플링 단계 수 (default: 50)
        cfg: Classifier-Free Guidance 설정 (default: 2.0)
        """
        b = y1.size(0)
        dt = 1.0 / sample_steps
        dt = torch.tensor([dt] * b).to(y1.device).view([b, *([1] * len(y1.shape[1:]))])

        z = y1
        for i in range(1, sample_steps + 1):
            t = i / sample_steps
            t = torch.tensor([t] * b).to(y1.device)

            vc = self.model(z, t, cond)  # 조건을 적용한 velocity field 예측
            if null_cond is not None:
                vu = self.model(z, t, null_cond)
                vc = vu + cfg * (vc - vu)

            z = z + dt * vc  # 역방향으로 진행

        return z

    


if __name__ == "__main__":
    # train class conditional RF on mnist.
    import numpy as np
    import torch.optim as optim
    from PIL import Image
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from torchvision.utils import make_grid
    from tqdm import tqdm

    import wandb
    from dit import DiT_Llama

    # add
    from torch.utils.data import Subset
    import copy

    parser = argparse.ArgumentParser(description="use cifar?")
    parser.add_argument("--cifar", action="store_true")
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

    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {model_size}, {model_size / 1e6}M")

    rf = RF(model)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    criterion = torch.nn.MSELoss()

    mnist = fdatasets(root="./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(mnist, batch_size=64, shuffle=True, drop_last=True)

    wandb.init(project=f"rf_{dataset_name}")

    ####################################################################################

    # Train RF
    for epoch in range(10):
        lossbin = {i: 0 for i in range(10)}
        losscnt = {i: 1e-6 for i in range(10)}
        for i, (x, c) in tqdm(enumerate(dataloader)):
            x, c = x.cuda(), c.cuda()
            optimizer.zero_grad()
            loss, blsct = rf.forward(x, c)
            loss.backward()
            optimizer.step()

            wandb.log({"loss": loss.item()})

            # count based on t
            for t, l in blsct:
                lossbin[int(t * 10)] += l
                losscnt[int(t * 10)] += 1

        # log
        for i in range(10):
            print(f"Epoch: {epoch}, {i} range loss: {lossbin[i] / losscnt[i]}")

        wandb.log({f"lossbin_{i}": lossbin[i] / losscnt[i] for i in range(10)})

        rf.model.train()

    ####################################################################################

    # model parameters
    forget_threshold = 4.0  # δ from the paper
    error_bound = 0.15 * forget_threshold  # ε = 0.15 * δ
    alpha = 0.6  # Combination factor for losses, as indicated in the supplementary material
    gamma = 0.6  # Factor for KL-divergence
    learning_rate = 1e-4  # η

    # Split the dataset based on the labels
    forget_indices = [i for i, label in enumerate(mnist.targets) if label == 0]  # Forget images (label = 0)
    remember_indices = [i for i, label in enumerate(mnist.targets) if label != 0]  # Remember images (label != 0)

    # Create datasets for forget and remember
    forget_dataset = Subset(mnist, forget_indices)
    remember_dataset = Subset(mnist, remember_indices)

    # Create DataLoader for both forget and remember datasets
    forget_dataloader = DataLoader(forget_dataset, batch_size=64, shuffle=True, drop_last=True)
    remember_dataloader = DataLoader(remember_dataset, batch_size=64, shuffle=True, drop_last=True)

    # Printing dataset sizes for verification
    print(f"Number of Forget Images: {len(forget_indices)}")
    print(f"Number of Remember Images: {len(remember_indices)}")

    ####################################################################################


    ####################################################################################

    # Function to compute the negative log-likelihood (NLL) for a batch
    def compute_nll(model, x_batch, c_batch):
        x0 = model.reverse_sample(x_batch, c_batch)
        x1 = x_batch

        # (256, 1, 32, 32) -> (256, 32*32)로 reshape
        x0_flat = x0.view(x_batch.shape[0], -1)
        x1_flat = x1.view(x_batch.shape[0], -1)

        # x0와 x1의 차이 계산
        diff = x0_flat - x1_flat

        # 차이에 대한 L2 norm 계산 (dim=1을 기준으로 계산)
        l2_norm_diff = torch.norm(diff, p=2, dim=1)

        return l2_norm_diff




    # dµR,σR (XF, δ; θT) calculation
    def d_function(nll_f, mu_R, sigma_R, delta):
        """
        Calculate dµR,σR (XF, δ; θT) as described in the paper.
        """
        d = (nll_f - (mu_R + delta * sigma_R)) / sigma_R
        return d

    # Forgetting loss LF
    def forget_loss(nll_f, mu_R, sigma_R, delta):
        """
        Compute the forget loss LF as described in the paper.
        """
        d = d_function(nll_f, mu_R, sigma_R, delta)
        return torch.sigmoid(d ** 2).mean()  # Using the sigmoid of squared d

    # Remembering loss LR (combines MSE loss and KL divergence losses)
    def remember_loss(model_T, model_B, x_r, c_r):
        """
        Compute the remembering loss LR.
        """
        # Negative log-likelihood of the remember set (X_R)
        nll_r_T = compute_nll(model_T, x_r, c_r)
        nll_r_B = compute_nll(model_B, x_r, c_r)

        # Forward and reverse KL divergence losses
        kl_forward = (nll_r_T * torch.log(nll_r_T / (nll_r_B + 1e-8))).mean()  # Forward KL divergence
        kl_reverse = (nll_r_B * torch.log(nll_r_B / (nll_r_T + 1e-8))).mean()  # Reverse KL divergence

        # Total loss
        return (1 - gamma) * criterion(nll_r_T, nll_r_B) + gamma * (kl_forward + kl_reverse)
    
    ####################################################################################

    # Fine-tuning

    # Step 1: θT ← θB, creates a deep copy of the model parameters
    # theta_T = copy.deepcopy(rf)  # θT ← θB

    # Optimizer for the tamed model
    # lr = 5e-4
    optimizer = optim.Adam(model.parameters(), lr=5e-4)

    # Taming loop (iteration over batches)
    for iteration in range(100):  # Running for 800 iterations as per the supplementary material
        # Sample batches from forget (DF) and remember (DR) datasets
        for (x_f, c_f), (x_r, c_r) in zip(forget_dataloader, remember_dataloader):
            # Move data to GPU
            x_f, c_f = x_f.cuda(), c_f.cuda()
            x_r, c_r = x_r.cuda(), c_r.cuda()


            ###########################################################################

            # # Step 4: Estimate the distribution (µ_R, σ_R) ← - log p_θT(XR)
            # nll_r = compute_nll(theta_T, x_r, c_r)
            # nll_r.requires_grad_(True)
            # mu_R = nll_r.mean().item()  # Mean of NLL
            # sigma_R = nll_r.std().item()  # Standard deviation of NLL

            # # Step 5: Calculate d = d_(µ_R, σ_R) (X_F , δ; θ_T)
            # nll_f = compute_nll(theta_T, x_f, c_f)
            # nll_f.requires_grad_(True)
            # d = d_function(nll_f, mu_R, sigma_R, forget_threshold)

            # # Step 6: Check if ∀i : | d_i | < ε (error bound condition)
            # if torch.all(torch.abs(d) < error_bound):
            #     print(f"Stopping early at iteration {iteration} as condition is met.")
            #     break

            # # Step 8: Compute total loss L
            # L_F = forget_loss(nll_f, mu_R, sigma_R, forget_threshold)
            # L_R = remember_loss(theta_T, rf, x_r, c_r)
            # total_loss = alpha * L_F + (1 - alpha) * L_R


            ###########################################################################

            nll_r = compute_nll(rf, x_r, c_r)
            nll_r.requires_grad_(True)
            mu_R = nll_r.mean()
            sigma_R = nll_r.std()

            nll_f = compute_nll(rf, x_f, c_f)
            nll_f.requires_grad_(True)
            mu_F = nll_f.mean()
            sigma_F = nll_f.std()

            # loss = mu_R - 2 * mu_F + 50
            loss = mu_R * mu_R / mu_F



            # total_loss.requires_grad_(True)

            # print(f"Total Loss grad_fn: {total_loss.grad_fn}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print(loss)

        # Print the progress of the iteration
        print(f"Iteration {iteration}, Loss: {loss.item()}")

        rf.model.train()

    # Step 10: Return the tamed model f_(θ_T)
    # tamed_model = theta_T
    print("Taming process complete.")

####################################################################################


    for epoch in range(10):
        rf.model.eval()
        with torch.no_grad():
            cond = torch.arange(0, 16).cuda() % 10
            uncond = torch.ones_like(cond) * 10

            init_noise = torch.randn(16, channels, 32, 32).cuda()
            images = rf.sample(init_noise, cond, uncond)
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

        rf.model.train()
