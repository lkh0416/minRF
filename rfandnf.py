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

class RF_tame:
    def __init__(self, model_taming, model, ln=True, th=0.15, dis=4.0):
        self.model_taming = model_taming
        self.model = model
        self.ln = ln
        self.th = th
        self.dis = dis

    def forward(self, x_forget, c_forget, x_remember, c_remember):
        # Estimate distribution on x_remember and model_taming
        ## print(x_forget.shape) # torch.Size([256, 3, 32, 32])
        null_cond = torch.ones_like(c_remember) * 10
        z = torch.randn_like(x_remember)
        b = x_remember.size(0)
        sample_steps = 1
        dt = 1.0 / sample_steps
        dt = torch.tensor([dt] * b).to(x_remember.device).view([b, *([1] * len(x_remember.shape[1:]))])
        for i in range(sample_steps, 0, -1):
            t = i / sample_steps
            t = torch.tensor([t] * b).to(x_remember.device)

            vc = self.model_taming(z, t, c_remember)
            # if null_cond is not None:
            #     vu = self.model_taming(z, t, null_cond)
            #     vc = vu + self.dis * (vc - vu)
            z = z - dt * vc

        print(z.shape)
        log_probs = F.log_softmax(z, dim=1)
        print(log_probs.shape)
        print(c_remember.shape)
        true_log_probs = log_probs.gather(dim=1, index=c_remember.unsqueeze(1))
        nll = -true_log_probs.mean()

        print(nll)

        # calculate distance

        # under threshold, break

        # Calculate forgetting loss

        # Calculate remembering loss

        # Calculate total loss

        return 0, 0, 0 # loss, loss_forget, loss_remember
        
        # z1 = torch.randn_like(x)
        # zt = (1 - texp) * x + texp * z1
        # vtheta = self.model(zt, t, cond)
        # batchwise_mse = ((z1 - x - vtheta) ** 2).mean(dim=list(range(1, len(x.shape))))
        # tlist = batchwise_mse.detach().cpu().reshape(-1).tolist()
        # ttloss = [(tv, tloss) for tv, tloss in zip(t, tlist)]
        # return batchwise_mse.mean(), ttloss

    @torch.no_grad()
    def sample(self, z, cond, null_cond=None, sample_steps=50, cfg=2.0):
        b = z.size(0)
        dt = 1.0 / sample_steps
        dt = torch.tensor([dt] * b).to(z.device).view([b, *([1] * len(z.shape[1:]))])
        images = [z]
        for i in range(sample_steps, 0, -1):
            t = i / sample_steps
            t = torch.tensor([t] * b).to(z.device)

            vc = self.model_taming(z, t, cond)
            if null_cond is not None:
                vu = self.model_taming(z, t, null_cond)
                vc = vu + cfg * (vc - vu)

            z = z - dt * vc
            images.append(z)
        return images

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
    # else:
    #     dataset_name = "mnist"
    #     fdatasets = datasets.MNIST
    #     transform = transforms.Compose(
    #         [
    #             transforms.ToTensor(),
    #             transforms.Pad(2),
    #             transforms.Normalize((0.5,), (0.5,)),
    #         ]
    #     )
    #     channels = 1
    #     model = DiT_Llama(
    #         channels, 32, dim=64, n_layers=6, n_heads=4, num_classes=10
    #     ).cuda()

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
    forget_class_loader = DataLoader(forget_class_dataset, batch_size=256, shuffle=True)
    remember_class_loader = DataLoader(remember_class_dataset, batch_size=256, shuffle=True)

    # load pretrained model for cifar 'model.pt'
    model.load_state_dict(torch.load('model.pt'))

    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {model_size}, {model_size / 1e6}M")

    # fixing model parameters of base model
    model_taming = copy.deepcopy(model)
    
    for param in model.parameters():
        param.requires_grad = False

    rf_tame = RF_tame(model_taming, model, th=args.th)

    wandb.init(project=f"rf_taming_{dataset_name}")

    optimizer = optim.SGD(model_taming.parameters(), lr=5e-4)
    # criterion = torch.nn.MSELoss()

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
            loss, loss_forget, loss_remember = rf_tame.forward(x_forget, c_forget, x_remember, c_remember)
            optimizer.step()

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

    #     rf.model.eval()
    #     with torch.no_grad():
    #         cond = torch.arange(0, 16).cuda() % 10
    #         uncond = torch.ones_like(cond) * 10

    #         init_noise = torch.randn(16, channels, 32, 32).cuda()
    #         images = rf.sample(init_noise, cond, uncond)
    #         # image sequences to gif
    #         gif = []
    #         for image in images:
    #             # unnormalize
    #             image = image * 0.5 + 0.5
    #             image = image.clamp(0, 1)
    #             x_as_image = make_grid(image.float(), nrow=4)
    #             img = x_as_image.permute(1, 2, 0).cpu().numpy()
    #             img = (img * 255).astype(np.uint8)
    #             gif.append(Image.fromarray(img))

    #         gif[0].save(
    #             f"contents/sample_{epoch}.gif",
    #             save_all=True,
    #             append_images=gif[1:],
    #             duration=100,
    #             loop=0,
    #         )

    #         last_img = gif[-1]
    #         last_img.save(f"contents/sample_{epoch}_last.png")

    #     rf.model.train()