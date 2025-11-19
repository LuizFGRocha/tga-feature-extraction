import torch
from torch import nn

class ConvAutoencoder(nn.Module):
    """
    A 1D Convolutional Autoencoder.
    This model is much more powerful than the SimpleAutoencoder as it
    respects the 1D structure of the data (ch_in=2, length=1024).
    It uses parameter sharing via convolutions, making it more
    efficient and a better choice for small datasets than a large
    fully-connected model.
    """
    def __init__(self, ch_in=2, length=1024, compressed_dim=3):
        super(ConvAutoencoder, self).__init__()
        
        self.input_channels = ch_in
        self.input_length = length
        
        # --- Encoder ---
        # We will downsample the length (1024) five times
        # 1024 -> 512 -> 256 -> 128 -> 64 -> 32
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(ch_in, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2) # (8, 512)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2) # (16, 256)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2) # (32, 128)
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2) # (64, 64)
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2) # (64, 32)
        )
        
        # Calculate the flattened size after the encoder
        self.bottleneck_channels = 64
        self.bottleneck_length = length // (2**5) # 1024 / 32 = 32
        self.bottleneck_flat_dim = self.bottleneck_channels * self.bottleneck_length
        
        # --- Bottleneck ---
        self.fc_compress = nn.Linear(self.bottleneck_flat_dim, compressed_dim)
        self.fc_decompress = nn.Linear(compressed_dim, self.bottleneck_flat_dim)

        # --- Decoder ---
        # We will upsample five times
        # 32 -> 64 -> 128 -> 256 -> 512 -> 1024
        
        self.upconv1 = nn.Sequential(
            nn.Upsample(scale_factor=2), # (64, 64)
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.upconv2 = nn.Sequential(
            nn.Upsample(scale_factor=2), # (32, 128)
            nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.upconv3 = nn.Sequential(
            nn.Upsample(scale_factor=2), # (16, 256)
            nn.Conv1d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.upconv4 = nn.Sequential(
            nn.Upsample(scale_factor=2), # (8, 512)
            nn.Conv1d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.upconv5 = nn.Sequential(
            nn.Upsample(scale_factor=2), # (8, 1024)
            nn.Conv1d(8, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Final layer to restore original channel count
        self.final_conv = nn.Conv1d(8, self.input_channels, kernel_size=1, stride=1, padding=0)

        
    def encode(self, x):
        # 1. Pass through encoder
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x) # Shape: (batch, 64, 32)
        
        # 2. Flatten
        x = x.view(x.size(0), -1) # Shape: (batch, 64 * 32 = 2048)
        
        # 3. Compress
        compressed = self.fc_compress(x) # Shape: (batch, 3)
        return compressed

    def forward(self, x):
        # 1. Encode
        compressed = self.encode(x) # Shape: (batch, 3)
        
        # 2. Decompress
        d = self.fc_decompress(compressed) # Shape: (batch, 2048)
        
        # 3. Reshape for decoder
        d = d.view(x.size(0), self.bottleneck_channels, self.bottleneck_length) # Shape: (batch, 64, 32)
        
        # 4. Pass through decoder
        d = self.upconv1(d)
        d = self.upconv2(d)
        d = self.upconv3(d)
        d = self.upconv4(d)
        d = self.upconv5(d)
        
        # 5. Final output
        reconstruction = self.final_conv(d) # Shape: (batch, 2, 1024)
        
        return reconstruction

if __name__ == '__main__':
    # Example usage:
    # Input shape: (batch_size, channels, length)
    # Batch of 4 samples, 2 channels, 1024 length
    test_input = torch.randn(4, 2, 1024)
    
    # Initialize the model
    model = ConvAutoencoder(ch_in=2, length=1024, compressed_dim=3)
    
    # Test the forward pass (reconstruction)
    reconstruction = model(test_input)
    
    # Test the encode pass
    encoding = model.encode(test_input)
    
    print(f"--- ConvAutoencoder Test ---")
    print(f"Original input shape: {test_input.shape}")
    print(f"Reconstruction shape: {reconstruction.shape}")
    print(f"Encoding shape:       {encoding.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    print("\nThis model is much more powerful and parameter-efficient.")