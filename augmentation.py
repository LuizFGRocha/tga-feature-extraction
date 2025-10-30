import numpy as np
from torch.utils.data import Dataset
import os

class AugmentedTGADataset(Dataset):
    def __init__(self, DATA_DIR='./data', noise_std=0.01):
        """
        Args:
            DATA_DIR: Directory containing data.npz
            noise_std: Standard deviation of Gaussian noise to add
        """
        self.noise_std = noise_std
        data = np.load(os.path.join(DATA_DIR, 'data.npz'))['TGA']
        self.T = data[:, 0, :]
        self.W = data[:, 1, :]
        self.X = data[:, 1:3, :]
        
    def __len__(self):
        return self.X.shape[0]
    
    def add_noise(self, x):
        noise = np.random.normal(0, self.noise_std, x.shape)
        return x + noise
    
    def smooth_signal(self, x, window_size=3):
        if np.random.random() > 0.5:  # 50% chance to apply
            kernel = np.ones(window_size) / window_size
            return np.convolve(x, kernel, mode='same')
        return x
    
    def compute_derivative(self, W, T, normalize=True):
        dT = np.gradient(T)
        dW = np.gradient(W)
        dWdT = dW / dT
        
        if normalize:
            dWdT = -dWdT
            dWdT_min = dWdT.min()
            dWdT_max = dWdT.max()
            if dWdT_max > dWdT_min:
                dWdT = (dWdT - dWdT_min) / (dWdT_max - dWdT_min)
            else:
                dWdT = np.zeros_like(dWdT)
        
        return dWdT
    
    def compute_second_derivative(self, W, T, normalize=True):
        dWdT = self.compute_derivative(W, T, normalize=False)
        dT = np.gradient(T)
        
        # second derivative
        d_dWdT = np.gradient(dWdT)
        d2WdT2 = d_dWdT / dT
        
        if normalize:
            d2WdT2 = -d2WdT2
            d2WdT2_min = d2WdT2.min()
            d2WdT2_max = d2WdT2.max()
            if d2WdT2_max > d2WdT2_min:
                d2WdT2 = (d2WdT2 - d2WdT2_min) / (d2WdT2_max - d2WdT2_min)
            else:
                d2WdT2 = np.zeros_like(d2WdT2)
        
        return d2WdT2
    
    def __getitem__(self, idx):
        W_original = np.array(self.W[idx])
        T_original = np.array(self.T[idx])
        
        W_augmented = self.add_noise(W_original)
        W_augmented = self.smooth_signal(W_augmented)
        
        dWdT_augmented = self.compute_derivative(W_augmented, T_original)
        
        # Stack W and dW/dT
        x = np.stack([W_augmented, dWdT_augmented], axis=0)
        y = x.copy()
            
        return x, y


def create_augmented_dataset(original_data_path='./data/data.npz', 
                            output_path='./data/data_augmented.npz',
                            augmentation_factor=5):
    data = np.load(original_data_path)
    original_samples = data['samples']
    original_tga = data['TGA']
    
    augmented_samples = []
    augmented_tga = []
    
    dataset = AugmentedTGADataset(DATA_DIR='./data')
    
    for idx in range(len(dataset)):
        # Add original sample
        augmented_samples.append(original_samples[idx])
        augmented_tga.append(original_tga[idx])
        
        # Get temperature and original weight for this sample
        T_original = original_tga[idx, 0]
        W_original = original_tga[idx, 1]
        
        # Generate augmented versions
        for aug_idx in range(augmentation_factor):
            W_augmented = dataset.add_noise(W_original.copy())
            W_augmented = dataset.smooth_signal(W_augmented)
            
            # Recompute dW/dT and d²W/dT² from augmented W
            dWdT_augmented = dataset.compute_derivative(W_augmented, T_original)
            d2WdT2_augmented = dataset.compute_second_derivative(W_augmented, T_original)
            
            augmented_samples.append(f"{original_samples[idx]}_aug{aug_idx}")
            
            # Reconstruct full TGA format (T, W, dW/dT, d²W/d²T)
            full_sample = np.zeros((4, 1024))
            full_sample[0] = T_original  # Keep temperature unchanged
            full_sample[1] = W_augmented  # Augmented W
            full_sample[2] = dWdT_augmented  # Recomputed dW/dT
            full_sample[3] = d2WdT2_augmented  # Recomputed d²W/dT²
            augmented_tga.append(full_sample)
    
    np.savez(output_path,
             samples=np.array(augmented_samples),
             TGA=np.array(augmented_tga))
    
    print(f"Original dataset size: {len(original_samples)}")
    print(f"Augmented dataset size: {len(augmented_samples)}")
    print(f"Saved to: {output_path}")


def compare_derivatives(data_path='./data/data.npz', num_samples=5):
    import matplotlib.pyplot as plt
    
    data = np.load(data_path)
    TGA = data['TGA']
    samples = data['samples']
    
    indices = np.random.choice(len(TGA), min(num_samples, len(TGA)), replace=False)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, idx in enumerate(indices):
        T = TGA[idx, 0]
        W = TGA[idx, 1]
        dWdT_original = TGA[idx, 2]
        d2WdT2_original = TGA[idx, 3]
        
        temp_dataset = AugmentedTGADataset(DATA_DIR=data_path.replace('/data.npz', ''))
        
        # Reconstruct derivatives using the same method as augmentation
        dWdT_reconstructed = temp_dataset.compute_derivative(W, T, normalize=True)
        d2WdT2_reconstructed = temp_dataset.compute_second_derivative(W, T, normalize=True)
    
        # Plot W
        axes[i, 0].plot(T, W, 'b-', label='W', linewidth=1.5)
        axes[i, 0].set_xlabel('Temperature (T)')
        axes[i, 0].set_ylabel('Weight (W)')
        axes[i, 0].set_title(f'Sample: {samples[idx]}\nWeight vs Temperature')
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)
        
        # Plot dW/dT comparison
        axes[i, 1].plot(T, dWdT_original, 'b-', label='Original dW/dT', linewidth=1.5, alpha=0.7)
        axes[i, 1].plot(T, dWdT_reconstructed, 'r--', label='Reconstructed dW/dT', linewidth=1.5, alpha=0.7)
        axes[i, 1].set_xlabel('Temperature (T)')
        axes[i, 1].set_ylabel('dW/dT')
        axes[i, 1].set_title(f'First Derivative Comparison')
        axes[i, 1].legend()
        axes[i, 1].grid(True, alpha=0.3)
        
        # Plot d²W/dT² comparison
        axes[i, 2].plot(T, d2WdT2_original, 'b-', label='Original d²W/dT²', linewidth=1.5, alpha=0.7)
        axes[i, 2].plot(T, d2WdT2_reconstructed, 'r--', label='Reconstructed d²W/dT²', linewidth=1.5, alpha=0.7)
        axes[i, 2].set_xlabel('Temperature (T)')
        axes[i, 2].set_ylabel('d²W/dT²')
        axes[i, 2].set_title(f'Second Derivative Comparison')
        axes[i, 2].legend()
        axes[i, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/derivative_comparison.png', dpi=150, bbox_inches='tight')
    
    return fig

if __name__ == "__main__":
    compare_derivatives(data_path='./data/data.npz', num_samples=3)
