import torch.nn as nn
import torch


class VAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=2, padding=1),  # 28*28 -> 14*14
            nn.ReLU(),
            nn.Conv2d(4, 16, kernel_size=3, stride=2, padding=1),  # 14*14 -> 7*7
            nn.ReLU(),
            nn.Flatten()  # (B, 32, 7, 7) -> (B, 1568)
        )

        self.mean = nn.Linear(16*7*7, 20)
        self.log_variance = nn.Linear(16*7*7, 20)
        self.decoder_fc = nn.Sequential(
            nn.Linear(20, 16*7*7),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def reparameterization(self, mean, log_variance):  # To have a distribution rather than a single point
        standard_deviation = torch.exp(0.5*log_variance)  # Std of our normal distribution for a layer
        epsilon = torch.randn_like(standard_deviation)  # Noise
        return mean + standard_deviation*epsilon  # We here return a kind of distribution for good latent space

    def forward(self, x):
        encoded_image = self.encoder(x)
        mean = self.mean(encoded_image)
        log_variance = self.log_variance(encoded_image)
        reparameterized_space = self.reparameterization(mean, log_variance)
        dfc = self.decoder_fc(reparameterized_space)
        decoder_input = dfc.view(-1, 16, 7, 7)
        return self.decoder(decoder_input), mean, log_variance
