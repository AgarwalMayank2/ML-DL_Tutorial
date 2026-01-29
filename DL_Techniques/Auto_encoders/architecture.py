import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            # nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1),  # 28*28 -> 28*28
            nn.Conv2d(1, 2, kernel_size=3, stride=2, padding=1),  # 28*28 -> 14*14
            nn.ReLU(),
            # nn.Conv2d(2, 4, kernel_size=3, stride=1, padding=1),  # 14*14 -> 14*14
            nn.Conv2d(2, 4, kernel_size=3, stride=2, padding=1),  # 14*14 -> 7*7
            nn.ReLU(),
            # nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1),  # 7*7 -> 7*7
            nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1),  # 7*7 -> 4*4
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            # nn.ConvTranspose2d(8, 4, kernel_size=3, stride=1, padding=1),  # 4*4 -> 4*4
            nn.ConvTranspose2d(8, 4, kernel_size=3, stride=2, padding=1),  # 4*4 -> 7*7
            nn.ReLU(),
            # nn.ConvTranspose2d(4, 2, kernel_size=3, stride=1, padding=1),  # 7*7 -> 7*7
            nn.ConvTranspose2d(4, 2, kernel_size=3, stride=2, padding=1, output_padding=1),  # 7*7 -> 14*14
            nn.ReLU(),
            # nn.ConvTranspose2d(2, 1, kernel_size=3, stride=1, padding=1),  # 14*14 -> 14*14
            nn.ConvTranspose2d(2, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # 14*14 -> 28*28
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)
