import numpy as np
import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, input):  # noqa: ANN001, ANN201, A002
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self, C, H, W):  # noqa: ANN001, ANN204
        super().__init__()
        self.C = C
        self.H = H
        self.W = W

    def forward(self, input):  # noqa: ANN201, ANN001, A002
        # input shape is (batch_size, C*H*W)
        return input.view(input.size(0), self.C, self.H, self.W)


class VAE(nn.Module):
    def __init__(
        self,
        image_channels: int = 4,
        h_dim: int = 4608,  # 512 channels * 3x3 spatial = 4608
        z_dim: int = 2304,
    ) -> None:  # Target latent dimension
        super(VAE, self).__init__()  # noqa:  UP008

        # ------------------------
        #       Encoder
        # ------------------------
        self.encoder = nn.Sequential(
            # Input: (4, 224, 224)
            nn.Conv2d(image_channels, 32, 4, 2, 1),  # 112x112
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2, 1),  # 56x56
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),  # 28x28
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),  # 14x14
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1),  # 7x7
            nn.LeakyReLU(0.2),
            # Adjusted layer to reach h_dim=4608 (512*3*3)
            nn.Conv2d(512, 512, 3, 2, 0),  # 3x3 spatial
            nn.LeakyReLU(0.2),
            Flatten(),  # Output: 512*3*3 = 4608
        )

        self.dropout = nn.Dropout(p=0.2)

        self.fc_mu = nn.Linear(h_dim, z_dim)
        self.fc_logvar = nn.Linear(h_dim, z_dim)
        self.fc_decode = nn.Linear(z_dim, h_dim)

        # ------------------------
        #       Decoder
        # ------------------------
        self.decoder = nn.Sequential(
            UnFlatten(512, 3, 3),  # Start from 512x3x3
            # Reverse encoder's final conv
            nn.ConvTranspose2d(512, 512, 3, 2, 0),  # 7x7
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # 14x14
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 28x28
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 56x56
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 112x112
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, image_channels, 4, 2, 1),  # 224x224
        )

    # Keep the reparameterize, encode, decode, and forward methods unchanged
    def reparameterize(self, mu, logvar):  # noqa: ANN201, ANN001
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):  # noqa: ANN201, ANN001
        h = self.encoder(x)
        h = self.dropout(h)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)  # noqa: F841
        return mu  # z, mu, logvar

    def decode(self, z):  # noqa: ANN201, ANN001
        h = self.fc_decode(z)
        return self.decoder(h)

    def forward(self, x):  # noqa: ANN201, ANN001
        z, mu, logvar = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


def generate_latent_representation(multi_channel: np.ndarray, vae: VAE):  # noqa: ANN201
    multi_channel = torch.from_numpy(multi_channel).unsqueeze(0).float()
    vae.eval()
    with torch.no_grad():
        ls = vae.encode(multi_channel)
    return ls
