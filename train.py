# handicl/train.py

from re import L
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import wandb
from pathlib import Path
import argparse
from tqdm import tqdm

from .dataset import HandICLDataset
from .handiclmodel import HandICL, mano_loss

def train(config):

    config_dict = vars(config)
    model_config = {
        'hidden_dim': config.hidden_dim,
        'num_heads': config.num_heads,
        'training': {
            'lr': config.learning_rate,
            'batch_size': config.batch_size,
            'num_epochs': config.num_epochs
        }
    }
    

    wandb.init(
        project="hand-icl",
        config=config_dict,
        name=config.run_name
    )


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    

    if config.json_path == 'all':
        json_list = [
        ]
        train_dataset = None
        for json_path in json_list:
            if train_dataset is None:
                train_dataset = HandICLDataset(
                    data_root=config.data_root,
                    transform=transform,
                    mask_ratio=config.mask_ratio,
                    json_path=json_path
                )
            else:
                train_dataset = torch.utils.data.ConcatDataset([
                    train_dataset,
                    HandICLDataset(
                        data_root=config.data_root,
                        transform=transform,
                        mask_ratio=config.mask_ratio,
                        json_path=json_path
                    )
                ])
            
    else:
        train_dataset = HandICLDataset(
            data_root=config.data_root,
            transform=transform,
            mask_ratio=config.mask_ratio,
            json_path=config.json_path
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    

    
    model = HandICL(model_config).to(device)


    

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.num_epochs,
        eta_min=config.min_lr
    )
    

    if config.resume:
        checkpoint = torch.load(config.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_loss = checkpoint['loss']
        current_lr = config.learning_rate
        print(f"Resuming from checkpoint at epoch {checkpoint['epoch']}")
    else:
        print("Starting from scratch")
    best_loss = float('inf')
    
    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.num_epochs}')
        for batch_idx, batch in enumerate(pbar):

            batch['img1'] = batch['img1'].to(device)
            batch['img2'] = batch['img2'].to(device)
            for k, v in batch['mano_params'].items():
                batch['mano_params'][k] = v.to(device)
            target = batch['target'].to(device)
            

            output = model(batch)
            loss = mano_loss(output, target)
            

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        

        avg_loss = epoch_loss / len(train_loader)
        

        wandb.log({
            'epoch': epoch,
            'loss': avg_loss,
            'learning_rate': current_lr
        })
        

        if avg_loss < best_loss:
            best_loss = avg_loss
            model_path = Path(config.save_dir) / f'{wandb.run.name}_best.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, model_path)
        

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Learning Rate: {current_lr:.6f}")
        print(f"Best Loss: {best_loss:.4f}")
        print("-" * 50)


    ckpt_path = Path(config.save_dir) / f'{wandb.run.name}_last.pth'
    torch.save({
        'epoch': config.num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }, ckpt_path)
    wandb.finish()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default='hand_icl_experiment')
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--json_path', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--mask_ratio', type=float, default=0.7)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')

    parser.add_argument('--hidden_dim', type=int, default=768)
    parser.add_argument('--num_heads', type=int, default=12)
    
    args = parser.parse_args()
    

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    
    train(args)
else:

    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default='hand_icl_experiment')
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--mask_ratio', type=float, default=0.7)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--log_interval', type=int, default=10)
    

    parser.add_argument('--hidden_dim', type=int, default=768)
    parser.add_argument('--num_heads', type=int, default=12)