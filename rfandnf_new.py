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
    forget_class_loader = DataLoader(forget_class_dataset, batch_size=32, shuffle=True)
    remember_class_loader = DataLoader(remember_class_dataset, batch_size=32, shuffle=True)

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
    rf_taminbg = RF(model_taming)

    wandb.init(project=f"rf_taming_new_{dataset_name}")

    optimizer = optim.SGD(model_taming.parameters(), lr=1e-3)
    nll_loss = torch.nn.NLLLoss()

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
            
            # Calculate forget loss

            # calculate remember loss
            
            loss.backward()
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

        rf.model_taming.eval()
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

        rf.model_taming.train()