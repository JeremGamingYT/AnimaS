import argparse
import os
from glob import glob
from PIL import Image
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class FrameDataset(Dataset):
    """Custom dataset that returns pairs of consecutive frames.

    Parameters
    ----------
    root_dir : str
        Directory containing frame images. Files must be named such that
        ``sorted(glob(root_dir/*.png))`` yields the chronological order.
    transform : callable, optional
        Optional transform to be applied on both input and target images.
    """

    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # Sort image paths alphanumerically
        self.paths = sorted(glob(os.path.join(root_dir, '*.png')))
        assert len(self.paths) >= 2, "Dataset must contain at least two images"

    def __len__(self):
        # We can form len(paths) - 1 pairs
        return len(self.paths) - 1

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        target_path = self.paths[idx + 1]
        # Load images
        img = Image.open(img_path).convert('RGB')
        target = Image.open(target_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
            target = self.transform(target)
        return img, target


def weights_init(m):
    """Initialize model weights as in the Pix2Pix paper."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class UNetDown(nn.Module):
    """Downsampling block: Conv -> BatchNorm -> LeakyReLU."""

    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        ]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    """Upsampling block: ConvTranspose -> BatchNorm -> Dropout -> ReLU."""

    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.ReLU(inplace=True))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        # Concatenate skip connection
        return torch.cat((x, skip_input), 1)


class GeneratorUNet(nn.Module):
    """U‑Net generator for frame prediction."""

    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.down1 = UNetDown(in_channels, 64, normalize=False)  # 128x128 -> 64x64
        self.down2 = UNetDown(64, 128)  # 64x64 -> 32x32
        self.down3 = UNetDown(128, 256)  # 32x32 -> 16x16
        self.down4 = UNetDown(256, 512)  # 16x16 -> 8x8
        self.down5 = UNetDown(512, 512)  # 8x8 -> 4x4
        self.down6 = UNetDown(512, 512)  # 4x4 -> 2x2
        self.down7 = UNetDown(512, 512)  # 2x2 -> 1x1

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )  # 1x1 -> 1x1 (no change as stride=2 and padding=1 on 1x1 will produce 1x1)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.final_up = nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1)
        self.final_activation = nn.Tanh()

    def forward(self, x):
        # Downsampling path with skip connections
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        b = self.bottleneck(d7)

        # Upsampling path
        u1 = self.up1(b, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        out = self.final_up(u7)
        return self.final_activation(out)


class PatchDiscriminator(nn.Module):
    """PatchGAN discriminator that classifies NxN patches as real or fake."""

    def __init__(self, in_channels=6):
        super().__init__()
        # The discriminator takes as input the concatenated condition and generated/real image
        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=2, padding=1, bias=False)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalize=False),  # 256x256 -> 128x128
            *discriminator_block(64, 128),  # 128x128 -> 64x64
            *discriminator_block(128, 256),  # 64x64 -> 32x32
            *discriminator_block(256, 512),  # 32x32 -> 16x16
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)  # 16x16 -> 15x15
        )

    def forward(self, img_A, img_B):
        # Concatenate condition and target/generator output along channel dimension
        x = torch.cat((img_A, img_B), 1)
        return self.model(x)


def save_sample(generator, dataloader, device, out_dir, num_samples=5):
    """Save a few sample predictions from the generator."""
    os.makedirs(out_dir, exist_ok=True)
    generator.eval()
    with torch.no_grad():
        for i, (img, target) in enumerate(dataloader):
            if i >= num_samples:
                break
            img = img.to(device)
            pred = generator(img).cpu()
            # Denormalize from [-1,1] to [0,1]
            img_np = (img + 1) / 2.0
            pred_np = (pred + 1) / 2.0
            target_np = (target + 1) / 2.0
            # Convert to PIL images
            to_pil = transforms.ToPILImage()
            for b in range(img_np.size(0)):
                concatenated = torch.cat((img_np[b], pred_np[b], target_np[b]), dim=2)
                pil_img = to_pil(concatenated)
                pil_img.save(os.path.join(out_dir, f'sample_{i}_{b}.png'))
            if i >= num_samples:
                break
    generator.train()


def train(opt):
    """Main training loop for the GAN."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data transforms: resize to 256x256, random crop, horizontal flip, and normalize to [-1,1]
    transform_list = [
        transforms.Resize((opt.img_size, opt.img_size), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    if opt.augment:
        transform_list = [
            transforms.Resize((opt.img_size + 30, opt.img_size + 30), Image.BICUBIC),
            transforms.RandomCrop((opt.img_size, opt.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    transform = transforms.Compose(transform_list)

    dataset = FrameDataset(opt.data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4, drop_last=True)

    # Initialize generator and discriminator
    generator = GeneratorUNet().to(device)
    discriminator = PatchDiscriminator().to(device)
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    # Loss functions
    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_L1 = nn.L1Loss()

    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    # Labels
    real_label = 1.0
    fake_label = 0.0

    for epoch in range(opt.epochs):
        for i, (real_A, real_B) in enumerate(dataloader):
            real_A = real_A.to(device)
            real_B = real_B.to(device)

            # Train Discriminator
            optimizer_D.zero_grad()
            # Real loss
            pred_real = discriminator(real_A, real_B)
            loss_D_real = criterion_GAN(pred_real, torch.full_like(pred_real, real_label, device=device))
            # Fake loss
            fake_B = generator(real_A)
            pred_fake = discriminator(real_A, fake_B.detach())
            loss_D_fake = criterion_GAN(pred_fake, torch.full_like(pred_fake, fake_label, device=device))
            # Total loss and backward
            loss_D = 0.5 * (loss_D_real + loss_D_fake)
            loss_D.backward()
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()
            # GAN loss (try to fool discriminator)
            pred_fake_for_G = discriminator(real_A, fake_B)
            loss_G_GAN = criterion_GAN(pred_fake_for_G, torch.full_like(pred_fake_for_G, real_label, device=device))
            # L1 loss
            loss_G_L1 = criterion_L1(fake_B, real_B) * opt.lambda_l1
            # Total generator loss
            loss_G = loss_G_GAN + loss_G_L1
            loss_G.backward()
            optimizer_G.step()

            if i % opt.log_interval == 0:
                print(f"[Epoch {epoch+1}/{opt.epochs}] [Batch {i+1}/{len(dataloader)}] "
                      f"[D loss: {loss_D.item():.4f}] [G loss: {loss_G.item():.4f}, "
                      f"adv: {loss_G_GAN.item():.4f}, L1: {loss_G_L1.item():.4f}]")

        # Save model checkpoints
        if (epoch + 1) % opt.checkpoint_interval == 0:
            os.makedirs(opt.checkpoint_dir, exist_ok=True)
            torch.save(generator.state_dict(), os.path.join(opt.checkpoint_dir, f"generator_epoch_{epoch+1}.pth"))
            torch.save(discriminator.state_dict(), os.path.join(opt.checkpoint_dir, f"discriminator_epoch_{epoch+1}.pth"))
            # Save sample predictions
            save_sample(generator, dataloader, device, opt.sample_dir, num_samples=5)


def parse_args():
    parser = argparse.ArgumentParser(description="Conditional GAN for next-frame prediction")
    parser.add_argument('--data_dir', type=str, required=True, help='Directory with input frames')
    parser.add_argument('--img_size', type=int, default=256, help='Size to resize images (square)')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate for optimizers')
    parser.add_argument('--lambda_l1', type=float, default=100.0, help='Weight of the L1 loss term')
    parser.add_argument('--augment', action='store_true', help='Use data augmentation (random crop/flip)')
    parser.add_argument('--checkpoint_interval', type=int, default=5, help='Save model every N epochs')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--sample_dir', type=str, default='samples', help='Directory to save example predictions')
    parser.add_argument('--log_interval', type=int, default=50, help='Interval between printing training progress')
    return parser.parse_args()


if __name__ == '__main__':
    # If this script is run directly, parse arguments and start training
    opt = parse_args()
    train(opt)