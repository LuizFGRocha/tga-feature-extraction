import torch
import torch.nn as nn

from models.base import TGAFeatureExtractor

class Bigger_VAE(TGAFeatureExtractor):
    def __init__(self, compressed_dim=64):
        super().__init__()
        
        # Shared Encoder layers
        self.conv_layers = nn.Sequential(
            nn.Conv1d(2, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # VAE specific heads
        self.fc_mu = nn.Linear(256 * 128, compressed_dim)
        self.fc_var = nn.Linear(256 * 128, compressed_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(compressed_dim, 256 * 128)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 2, 4, stride=2, padding=1)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        # For evaluation, we typically use the mean (mu) as the encoding
        x = self.conv_layers(x)
        return self.fc_mu(x)

    def forward(self, x):
        flat = self.conv_layers(x)
        mu = self.fc_mu(flat)
        logvar = self.fc_var(flat)
        
        z = self.reparameterize(mu, logvar)
        
        d = self.decoder_input(z)
        d = d.view(-1, 256, 128)
        reconstruction = self.decoder(d)
        
        # Return tuple for custom loss handling
        return reconstruction, mu, logvar

if __name__ == "__main__":
    model = VAE()
    print(f"Model parameter count: {sum(p.numel() for p in model.parameters())}")
