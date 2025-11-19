import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import os

# Import from your new structure
from models.factory import get_model
from src.dataset import TGADataset

def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training {args.model_name} on {device}")

    # 1. Setup Data
    dataset = TGADataset(data_path=args.data_path)
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = random_split(dataset, [train_size, test_size])
    
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size)

    # 2. Setup Model via Factory
    model = get_model(args.model_name, compressed_dim=args.latent_dim)
    model.double().to(device)

    # 3. Setup Loop
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    writer = SummaryWriter(f"runs/{args.model_name}_exp")
    
    os.makedirs(f"checkpoints/{args.model_name}", exist_ok=True)

    run_id = getattr(args, 'run_id', 'default')
    save_dir = f"checkpoints/{args.model_name}/{run_id}"
    os.makedirs(save_dir, exist_ok=True)
    
    final_path = ""

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for x, target in train_dl:
            x, target = x.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            global_step += 1

        # Logging
        writer.add_scalar('Loss/train', train_loss / len(train_dl), epoch)
        
        if epoch == args.epochs - 1: # Save at end
            final_path = os.path.join(save_dir, "final.pth")
            model.save_checkpoint(final_path, optimizer, epoch, train_loss)
            
    return final_path # <--- IMPORTANT: Return the path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="attention_unet or autoencoder")
    parser.add_argument("--data_path", type=str, default="./data/tga/data.npz")
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--save_interval", type=int, default=100)
    
    args = parser.parse_args()
    train(args)