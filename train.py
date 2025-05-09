
# This file is part of BCDDM and is released under the BSD 3-Clause License.
# 
# Copyright (c) 2025 Zelin Zhang, Ao Liu. All rights reserved.

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR, OneCycleLR, CosineAnnealingLR, ReduceLROnPlateau
import logging
import random
import math
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import argparse
from tqdm import tqdm
import json
from scipy import linalg
# Distributed training
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="BCDDM training script")
    # Basic training parameters
    parser.add_argument('--epochs', type=int, default=10000, help='The total number of training rounds')
    parser.add_argument('--batch_size', type=int, default=8, help='The number of samples per batch')
    parser.add_argument('--image_size', type=int, default=256, help='The size of the image (width and height)')
    parser.add_argument('--channels', type=int, default=1, help='The number of channels in the image')
    parser.add_argument('--time_steps', type=int, default=1000, help='The number of diffusion steps')
    parser.add_argument('--beta_start', type=float, default=0.0001, help='The starting value of β')
    parser.add_argument('--beta_end', type=float, default=0.02, help='The ending value of β')
    parser.add_argument('--label_dim', type=int, default=7, help='The dimension of the label')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--loss_noise_prop', type=float, default=0.8, help='The weight of the noise loss')
    parser.add_argument('--loss_label_prop', type=float, default=0.2, help='The weight of the label loss')
    parser.add_argument('--lr', type=float, default=0.005, help='The learning rate')
    parser.add_argument('--data_dir', type=str, default='RIAF_dataset_2157_files_npz', help='The directory of the dataset')
    parser.add_argument('--val_split', type=float, default=0.1, help='The split ratio of the validation set')
    parser.add_argument('--resume', type=str, default='', help='The path of the model to resume training')
    parser.add_argument('--save_every', type=int, default=100, help='Save the model every n epochs')
    parser.add_argument('--sample_after', type=int, default=1000, help='Generate samples after n epochs')
    # Add evaluation parameters
    parser.add_argument('--val_every', type=int, default=5, help='Evaluate every n epochs')
    parser.add_argument('--val_samples', type=int, default=-1, help='The number of samples used for evaluation, -1 means using all validation set')
    # Add learning rate scheduler parameters
    parser.add_argument('--lr_scheduler', type=str, default='step', choices=['step', 'cosine', 'onecycle', 'plateau'], help='The type of the learning rate scheduler')
    parser.add_argument('--warmup_epochs', type=int, default=0, help='The number of epochs for warmup')
    # Add ReduceLROnPlateau scheduler specific parameters
    parser.add_argument('--lr_factor', type=float, default=0.5, help='The learning rate decay factor (only used for plateau scheduler)')  
    parser.add_argument('--lr_patience', type=int, default=5, help='The number of evaluation cycles to wait before adjusting the learning rate (only used for plateau scheduler)')
    parser.add_argument('--lr_min', type=float, default=1e-6, help='The minimum learning rate (only used for plateau scheduler)')
    parser.add_argument('--lr_cooldown', type=int, default=2, help='The cooldown period after learning rate adjustment (only used for plateau scheduler)')
    parser.add_argument('--lr_freeze_epochs', type=int, default=0, help='The number of epochs to freeze before using the scheduler (only used for plateau scheduler)')
    # Add support for configuration files
    parser.add_argument('--config', type=str, default='', help='The path of the configuration file')
    # Add distributed training parameters
    parser.add_argument('--distributed', action='store_true', help='Whether to use distributed training')
    parser.add_argument('--local_rank', type=int, default=-1, help='The local rank in distributed training')
    parser.add_argument('--world_size', type=int, default=-1, help='The total number of processes in distributed training')
    parser.add_argument('--dist_url', type=str, default='env://', help='The URL for distributed training')
    parser.add_argument('--dist_backend', type=str, default='nccl', help='The backend for distributed training')
    
    args = parser.parse_args()
    
    # If a configuration file is specified, load parameters from the configuration file
    if args.config:
        if os.path.exists(args.config):
            with open(args.config, 'r') as f:
                config = json.load(f)

                # Training parameters
                if 'training' in config:
                    args.epochs = config['training'].get('epochs', args.epochs)
                    args.batch_size = config['training'].get('batch_size', args.batch_size)
                    args.lr = config['training'].get('lr', args.lr)
                    args.seed = config['training'].get('seed', args.seed)
                    args.save_every = config['training'].get('save_every', args.save_every)
                    args.sample_after = config['training'].get('sample_after', args.sample_after)
                    args.lr_scheduler = config['training'].get('lr_scheduler', args.lr_scheduler)
                    args.warmup_epochs = config['training'].get('warmup_epochs', args.warmup_epochs)
                    args.distributed = config['training'].get('distributed', args.distributed)
                
                # Model parameters
                if 'model' in config:
                    args.image_size = config['model'].get('image_size', args.image_size)
                    args.channels = config['model'].get('channels', args.channels)
                    args.time_steps = config['model'].get('time_steps', args.time_steps)
                    args.beta_start = config['model'].get('beta_start', args.beta_start)
                    args.beta_end = config['model'].get('beta_end', args.beta_end)
                    args.label_dim = config['model'].get('label_dim', args.label_dim)
                
                # Loss function parameters
                if 'loss' in config:
                    args.loss_noise_prop = config['loss'].get('noise_prop', args.loss_noise_prop)
                    args.loss_label_prop = config['loss'].get('label_prop', args.loss_label_prop)
                
                # Data parameters
                if 'data' in config:
                    args.data_dir = config['data'].get('data_dir', args.data_dir)
                    args.val_split = config['data'].get('val_split', args.val_split)

                # Evaluation parameters
                if 'evaluation' in config:
                    args.val_every = config['evaluation'].get('val_every', args.val_every)
                    args.val_samples = config['evaluation'].get('val_samples', args.val_samples)
                
                # Distributed training parameters
                if 'distributed' in config:
                    args.distributed = config['distributed'].get('enabled', args.distributed)
                    args.dist_backend = config['distributed'].get('backend', args.dist_backend)
                    args.dist_url = config['distributed'].get('url', args.dist_url)
                
                # Load ReduceLROnPlateau scheduler parameters
                if 'lr_scheduler_params' in config:
                    args.lr_factor = config['lr_scheduler_params'].get('factor', args.lr_factor)
                    args.lr_patience = config['lr_scheduler_params'].get('patience', args.lr_patience)
                    args.lr_min = config['lr_scheduler_params'].get('min_lr', args.lr_min)
                    args.lr_cooldown = config['lr_scheduler_params'].get('cooldown', args.lr_cooldown)
                    args.lr_freeze_epochs = config['lr_scheduler_params'].get('lr_freeze_epochs', args.lr_freeze_epochs)
                
                print(f"Configuration file '{args.config}' has been loaded")
        else:
            print(f"Configuration file '{args.config}' does not exist, using default values")
    
    return args

# Distributed training initialization function
def init_distributed_mode(args):
    """
    Initialize the distributed training environment
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.local_rank = args.rank % torch.cuda.device_count()
    else:
        print('No distributed training information found in environment variables')
        args.distributed = False
        return

    # If distributed training is used, set the relevant parameters
    if args.distributed:
        # Set the current device
        torch.cuda.set_device(args.local_rank)
        args.dist_backend = args.dist_backend
        print(f'| Distributed initialization: {args.dist_url}')
        # Initialize the process group
        dist.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url,
            world_size=args.world_size, rank=args.rank
        )
        # Synchronize all processes
        dist.barrier()
        # Set the distributed training flag
        args.is_master = (args.rank == 0)
    else:
        args.is_master = True

# Global parameter configuration
args = parse_args()

# Initialize the distributed environment (if enabled)
if args.distributed:
    init_distributed_mode(args)
else:
    # In non-distributed mode, set is_master to True by default
    args.is_master = True

# Set the device and hyperparameters
if args.distributed:
    device = torch.device(f"cuda:{args.local_rank}")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The total number of training rounds
num_epochs = args.epochs
# The number of samples per batch
batch_size = args.batch_size
# The size of the image (width and height)
image_size = args.image_size
# The number of channels in the image
channels = args.channels
# The number of diffusion steps
time_steps = args.time_steps
# The starting value of β (controls the rate of noise addition)
beta_start = args.beta_start
# The ending value of β
beta_end = args.beta_end
# The number of classes
num_classes = 1
# The dimension of the label
label_dim = args.label_dim
# The random seed
seed = args.seed
# The weight of the noise loss
loss_noise_prop = args.loss_noise_prop
# The weight of the label loss
loss_label_prop = args.loss_label_prop

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# Set the random seed
# seed_everything(seed)



class DoubleConv(nn.Module):
    """
    Double convolution module
    """
    def __init__(self, in_channels, out_channels):
        """
        Initialize the double convolution module
        :param in_channels: The number of input channels
        :param out_channels: The number of output channels
        """
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1), 
            nn.BatchNorm2d(out_channels),  
            nn.ReLU(inplace=True),  
            nn.Conv2d(out_channels, out_channels, 3, padding=1),  
            nn.BatchNorm2d(out_channels),  
            nn.ReLU(inplace=True) 
        )

    def forward(self, x):
        """
        Forward propagation function
        :param x: Input tensor
        :return: Feature map after double convolution processing
        """
        return self.conv(x)


# Conditional U-Net model
class ConditionalUNet(nn.Module):
    def __init__(self, out_ch=channels):
        """
        Conditional U-Net model
        """
        super().__init__()
        base_ch = 64  
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128)
        )

        # Label embedding
        self.label_embed = nn.Sequential(
            nn.Linear(label_dim, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128)
        )

        self.enc1 = DoubleConv(channels+128, base_ch)
        self.enc2 = DoubleConv(base_ch, base_ch*2)
        self.enc3 = DoubleConv(base_ch*2, base_ch*4)
        self.enc4 = DoubleConv(base_ch*4, base_ch*8)
        
        self.bridge = DoubleConv(base_ch*8, base_ch*16)

        self.dec4 = DoubleConv(base_ch*16, base_ch*8)
        self.dec3 = DoubleConv(base_ch*8, base_ch*4)
        self.dec2 = DoubleConv(base_ch*4, base_ch*2)
        self.dec1 = DoubleConv(base_ch*2, base_ch)

        self.pool = nn.MaxPool2d(2)

        self.upconv4 = nn.ConvTranspose2d(base_ch*16, base_ch*8, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(base_ch*8, base_ch*4, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(base_ch*4, base_ch*2, kernel_size=2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(base_ch*2, base_ch, kernel_size=2, stride=2)

        self.final = nn.Conv2d(base_ch, out_ch, kernel_size=1)

        # Label prediction branch
        self.predicted_labels = nn.Sequential(
            nn.Linear(base_ch*16*16*16, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, label_dim)
        )

    def forward(self, x, t, label):
        """
        Forward propagation
        """
        t_emb = self.time_embed(t.unsqueeze(-1).float())
        
        label_emb = self.label_embed(label)
        
        combined = t_emb + label_emb
        
        condition = combined.unsqueeze(-1).unsqueeze(-1)
        condition = condition.expand(-1, -1, x.shape[2], x.shape[3])
        
        x = torch.cat([x, condition], dim=1)
        
        enc1 = self.enc1(x)                  
        enc2 = self.enc2(self.pool(enc1))    
        enc3 = self.enc3(self.pool(enc2))    
        enc4 = self.enc4(self.pool(enc3))    
        bridge = self.bridge(self.pool(enc4))
        bridge_flatten = bridge.view(-1, 1024*16*16)
        predicted_labels = self.predicted_labels(bridge_flatten)

        dec4 = self.dec4(torch.cat([self.upconv4(bridge), enc4], dim=1))
        dec3 = self.dec3(torch.cat([self.upconv3(dec4), enc3], dim=1))  
        dec2 = self.dec2(torch.cat([self.upconv2(dec3), enc2], dim=1))  
        dec1 = self.dec1(torch.cat([self.upconv1(dec2), enc1], dim=1))  
        out = self.final(dec1) 
        
        return out, predicted_labels

# Define Diffusion Model
class DiffusionModel:
    def __init__(self):
        """
        Initialize Diffusion Model
        """
        self.model = ConditionalUNet().to(device)
        
        # If distributed training is used, wrap the model
        if args.distributed:
            self.model = DDP(
                self.model, 
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=False
            )
            
        self.betas = torch.linspace(beta_start, beta_end, time_steps).to(device)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def get_index_from_list(self, vals, t, x_shape):
        """
        Get the value at the specified time step from the alphas_cumprod list
        :param vals: alphas_cumprod list, (T,) 
        :param t: time step, (batch_size,)
        :param x_shape: output shape
        :return: value at the specified time step, shape same as x_shape
        """
        batch_size = t.shape[0]
        out = vals.gather(-1, t)
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device) # from (batch_size) to (batch_size,1,1,1)
        return out

    def forward_diffusion(self, x0, t):
        """
        Forward diffusion process
        :param x0: original image, shape (batch_size, channels, height, width)
        :param t: time step, shape (batch_size,)
        :return: noisy image and noise, shape same as x0
        """
        noise = torch.randn_like(x0)
        alphas_cumprod_t = self.get_index_from_list(self.alphas_cumprod, t, x0.shape)
        xt = torch.sqrt(alphas_cumprod_t) * x0 + torch.sqrt(1 - alphas_cumprod_t) * noise
        return xt, noise

    def train_step(self, x0, labels):
        """
        Training step
        :param x0: original image, shape (batch_size, channels, height, width)
        :param labels: image labels, shape (batch_size,)
        :return: loss value
        """
        t = torch.randint(0, time_steps, (x0.shape[0],), device=device).long()
        xt, noise = self.forward_diffusion(x0, t)
        predicted_noise, predicted_labels = self.model(xt, t / time_steps, labels)
        
        loss_noise = F.mse_loss(noise, predicted_noise)
        loss_label = F.mse_loss(labels, predicted_labels)
        
        # Calculate loss with weights
        loss = loss_noise_prop * loss_noise + loss_label_prop * loss_label
        
        return loss, loss_noise, loss_label

    @torch.no_grad()
    def sample(self, n_samples, labels, size=image_size):
        """
        Sample images from the model
        :param n_samples: number of generated images
        :param labels: labels, shape (n_samples,)
        :param size: size of the generated images
        :return: generated images, shape (n_samples, channels, size, size)
        """
        self.model.eval()
        x = torch.randn(n_samples, channels, size, size).to(device)

        for i in reversed(range(time_steps)):
            t = torch.full((n_samples,), i, device=device, dtype=torch.long)
            predicted_noise, _ = self.model(x, t / time_steps, labels)
            alpha = self.alphas[i]
            alpha_cumprod = self.alphas_cumprod[i]
            beta = self.betas[i]

            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * predicted_noise) + torch.sqrt(beta) * noise

        self.model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        return x

class ImageFileDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir: path to the folder containing images
        transform: optional transform to apply to images
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)

        data = np.load(img_path)
        image = np.array(data['I_rot'], dtype=np.float32)
        image = image / image.max()
        label = data['normalized_parm_array']
        label = torch.tensor(label, dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)

        return image, label

# 数据加载和预处理
def load_data(val_split=0.1):
    """
    Load BH dataset and preprocess, split into training and validation sets
    :param val_split: validation set ratio
    :return: training data loader and validation data loader
    """
    # Define image preprocessing
    # Set data transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    # Create custom dataset instance
    full_dataset = ImageFileDataset(root_dir=args.data_dir, transform=transform)
    
    # Calculate split size
    dataset_size = len(full_dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size
    
    # Randomly split dataset
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    if hasattr(args, 'is_master') and (args.is_master or not args.distributed):
        print(f"Dataset size: {dataset_size}, Training set: {train_size}, Validation set: {val_size}")
    elif not hasattr(args, 'is_master'):
        print(f"Dataset size: {dataset_size}, Training set: {train_size}, Validation set: {val_size}")
    
    # Create data loader
    if args.distributed:
        # Distributed training uses DistributedSampler
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=4,
            pin_memory=True
        )
    else:
        # Non-distributed training uses normal DataLoader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

# Save model parameters
os.makedirs('saved_models', exist_ok=True)
os.makedirs('create', exist_ok=True)
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_training.log'), 
        logging.StreamHandler()
    ]
)

@torch.no_grad()
def evaluate(model, dataloader, max_samples=-1):
    """
    Evaluate model performance
    :param model: Diffusion Model instance
    :param dataloader: data loader
    :param max_samples: maximum number of samples for evaluation, -1 means all samples
    :return: average loss
    """
    model.model.eval()
    total_loss = 0.0
    total_loss_noise = 0.0
    total_loss_label = 0.0
    num_batches = 0
    sample_count = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            batch_size = images.shape[0]
            
            if max_samples > 0 and sample_count + batch_size > max_samples:
                samples_needed = max_samples - sample_count
                images = images[:samples_needed]
                labels = labels[:samples_needed]
            
            t = torch.randint(0, time_steps, (images.shape[0],), device=device).long()
            
            xt, noise = model.forward_diffusion(images, t)
            
            predicted_noise, predicted_labels = model.model(xt, t / time_steps, labels)
            
            loss_noise = F.mse_loss(noise, predicted_noise)
            
            loss_label = F.mse_loss(labels, predicted_labels)
            
            loss = loss_noise_prop * loss_noise + loss_label_prop * loss_label
            
            total_loss += loss.item()
            total_loss_noise += loss_noise.item()
            total_loss_label += loss_label.item()
            num_batches += 1
            
            sample_count += images.shape[0]
            
            if max_samples > 0 and sample_count >= max_samples:
                break
    
    model.model.train()
    
    # Calculate average loss
    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    avg_loss_noise = total_loss_noise / num_batches if num_batches > 0 else float('inf')
    avg_loss_label = total_loss_label / num_batches if num_batches > 0 else float('inf')
    
    # Record the actual number of samples evaluated
    if args.is_master or not args.distributed:
        print(f"Evaluation completed: {sample_count} samples used for evaluation")
    
    return avg_loss, avg_loss_noise, avg_loss_label

def train(model, dataloader, num_epochs, start_epoch=0, val_dataloader=None):
    """
    Train Diffusion Model
    :param model: Diffusion Model instance
    :param dataloader: training data loader
    :param num_epochs: number of training epochs
    :param start_epoch: starting epoch (for resuming training)
    :param val_dataloader: validation data loader (optional)
    :return: current epoch, optimizer, and scheduler (for resuming training)
    """
    # Create TensorBoard logger
    writer = SummaryWriter(log_dir="runs")
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.model.parameters(), lr=args.lr)
    
    # Calculate total training steps
    total_steps = num_epochs * len(dataloader)

    # Record original learning rate, for warmup
    init_lr = args.lr
    
    # Add validation loss-based learning rate scheduler
    if args.lr_scheduler == 'plateau':
        # Use ReduceLROnPlateau based on validation loss to adjust learning rate
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='min',                # Monitor the minimum value of validation loss
            factor=args.lr_factor,     # Learning rate multiplied by factor
            patience=args.lr_patience, # Adjust learning rate when validation loss stops improving after n evaluation cycles
            min_lr=args.lr_min,        # Minimum learning rate
            cooldown=args.lr_cooldown  # Cooldown period after adjusting learning rate
        )
        use_val_loss_scheduler = True
    # step learning rate scheduler
    elif args.lr_scheduler == 'step':
        # Use warmup_epochs to implement warmup
        if args.warmup_epochs > 0:
            # Create a learning rate scheduler with warmup followed by stepwise decay
            warmup_steps = args.warmup_epochs * len(dataloader)
            def lr_lambda(current_step):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                else:
                    # Stepwise decay every 1000 steps
                    return 0.98 ** ((current_step - warmup_steps) // 1000)
            
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        else:
            scheduler = StepLR(optimizer, step_size=1000, gamma=0.98)
        use_val_loss_scheduler = False
    elif args.lr_scheduler == 'cosine':
        # Use warmup_epochs to implement warmup
        if args.warmup_epochs > 0:
            warmup_steps = args.warmup_epochs * len(dataloader)
            def lr_lambda(current_step):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                else:
                    # Cosine decay part
                    progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
            
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
        use_val_loss_scheduler = False
    elif args.lr_scheduler == 'onecycle':
        # OneCycleLR starts from low learning rate, then rises and falls
        # If warmup_epochs is set, use it to determine pct_start
        if args.warmup_epochs > 0:
            pct_start = min(0.5, float(args.warmup_epochs) / float(num_epochs))
        else:
            pct_start = 0.3  # Default value
            
        scheduler = OneCycleLR(
            optimizer, 
            max_lr=args.lr,
            total_steps=total_steps,
            pct_start=pct_start,  # The ratio of the learning rate rise phase to the total training steps
            div_factor=25.0,  # Initial learning rate = max_lr / div_factor
            final_div_factor=1000.0  # Final learning rate = max_lr / final_div_factor
        )
        use_val_loss_scheduler = False
    
    # If there is a checkpoint to resume, load the optimizer and scheduler state
    if args.resume and os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume)
        if isinstance(checkpoint, dict):
            if 'optimizer_state_dict' in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    print("Optimizer state restored")
                except Exception as e:
                    print(f"Failed to restore optimizer state: {e}")
            
            if 'scheduler_state_dict' in checkpoint:
                try:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    print("Scheduler state restored")
                except Exception as e:
                    print(f"Failed to restore scheduler state: {e}")
    
    global_step = 0
    best_loss = float('inf')
    
    # Current epoch value (for saving during interruption)
    current_epoch = start_epoch
    
    try:
        for epoch in range(start_epoch, num_epochs):
            current_epoch = epoch  # Update current epoch

            # Add learning rate warmup for ReduceLROnPlateau
            if args.lr_scheduler == 'plateau' and args.warmup_epochs > 0 and epoch < args.warmup_epochs:
                # Linear warmup learning rate
                warmup_factor = float(epoch + 1) / float(max(1, args.warmup_epochs))
                new_lr = init_lr * warmup_factor
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
                if args.is_master or not args.distributed:
                    print(f"Warmup phase ({epoch+1}/{args.warmup_epochs}): Learning rate set to {new_lr:.6f}")
            
            # Training phase
            model.model.train()
            epoch_loss = 0.0
            epoch_loss_noise = 0.0
            epoch_loss_label = 0.0
            num_batches = 0
            
            # Show progress
            for batch, (images, labels) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)):
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                loss, loss_noise, loss_label = model.train_step(images, labels)
                loss.backward()
                optimizer.step()
                
                # If using OneCycleLR or custom learning rate scheduler, update learning rate every batch
                if not use_val_loss_scheduler and (args.lr_scheduler == 'onecycle' or (args.warmup_epochs > 0 and (args.lr_scheduler == 'step' or args.lr_scheduler == 'cosine'))):
                    scheduler.step()
                
                # Accumulate loss values for calculating average loss per epoch
                epoch_loss += loss.item()
                epoch_loss_noise += loss_noise.item()
                epoch_loss_label += loss_label.item()
                num_batches += 1
                
                # Record loss values for each batch to TensorBoard
                writer.add_scalar('Loss/batch/total', loss.item(), global_step)
                writer.add_scalar('Loss/batch/noise', loss_noise_prop * loss_noise.item(), global_step)
                writer.add_scalar('Loss/batch/label', loss_label_prop * loss_label.item(), global_step)
                writer.add_scalar('LR', optimizer.param_groups[0]['lr'], global_step)
                
                global_step += 1
            
            # Output log at the end of each epoch
            if num_batches > 0:
                avg_loss = epoch_loss / num_batches
                avg_loss_noise = epoch_loss_noise / num_batches
                avg_loss_label = epoch_loss_label / num_batches
                
                logging.info(
                    f"Epoch {epoch+1}/{num_epochs} - "
                    f"Loss: {avg_loss:.6f} - "
                    f"Noise Loss: {loss_noise_prop * avg_loss_noise:.6f} - "
                    f"Label Loss: {loss_label_prop * avg_loss_label:.6f} - "
                    f"LR: {optimizer.param_groups[0]['lr']:.6f}"
                )
            
            # If the learning rate scheduler is not based on validation loss, and is not OneCycleLR or custom learning rate scheduler, update learning rate every epoch
            if not use_val_loss_scheduler and not (args.lr_scheduler == 'onecycle' or (args.warmup_epochs > 0 and (args.lr_scheduler == 'step' or args.lr_scheduler == 'cosine'))):
                scheduler.step()
            
            # Record average loss for each epoch to TensorBoard
            if num_batches > 0:
                avg_train_loss = epoch_loss / num_batches
                writer.add_scalar('Loss/epoch/train/total', avg_train_loss, epoch)
                writer.add_scalar('Loss/epoch/train/noise', epoch_loss_noise / num_batches * loss_noise_prop, epoch)
                writer.add_scalar('Loss/epoch/train/label', epoch_loss_label / num_batches * loss_label_prop, epoch)

                
            #     print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.6f}")

            # Evaluation phase (if there is a validation set)
            if val_dataloader and epoch % args.val_every == 0:  # Evaluate every args.val_every epochs
                val_loss, val_loss_noise, val_loss_label = evaluate(model, val_dataloader, args.val_samples)
                writer.add_scalar('Loss/epoch/val/total', val_loss, epoch)
                writer.add_scalar('Loss/epoch/val/noise', val_loss_noise * loss_noise_prop, epoch)
                writer.add_scalar('Loss/epoch/val/label', val_loss_label * loss_label_prop, epoch)
                
                print(f"Epoch {epoch+1} - Validation Loss: {val_loss:.6f}")
                
                # If using a learning rate scheduler based on validation loss, update learning rate using validation loss
                if use_val_loss_scheduler:
                    # Check if in warmup or freezing period
                    if args.warmup_epochs > 0 and epoch < args.warmup_epochs:
                        # Warmup phase does not update scheduler
                        print(f"During the preheating period ({epoch+1}/{args.warmup_epochs}), skip scheduler step")
                    elif args.lr_freeze_epochs > 0 and epoch < args.lr_freeze_epochs:
                        # Freezing period does not update scheduler
                        print(f"During the learning rate freezing period ({epoch+1}/{args.lr_freeze_epochs}), skip scheduler step")
                    else:
                        # Normal update scheduler
                        scheduler.step(val_loss)
                    
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"Current learning rate: {current_lr:.8f}")
                
                # Save best model
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_model_path = os.path.join("saved_models", "best_model.pth")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': best_loss,
                        'args': vars(args)
                    }, best_model_path)
                    print(f"Save the best model, loss: {best_loss:.6f}")

            # After each epoch, generate some samples
            if (epoch + 1) % args.save_every == 0 and (epoch+1) >= args.sample_after:
                # Define fixed parameter values
                num1 = torch.tensor([-0.15812528315002394,-0.23035591572195277,-0.7324235022937507,1.2727545262149988,-0.8974121127363188,1.0,1.4529981774605583]).to(device)
                num2 = torch.tensor([-1.1057298731294203,-0.21688251056084387,1.4097393802702631,-1.4395498496223589,0.6645476296183156,-1.0,1.3663956370821144]).to(device)
                num3 = torch.tensor([-1.6237791582331897,1.5923921135901375,-0.10340549661170463,-0.8820251347685134,-0.5607514115058709,1.0,-0.7890453678924886]).to(device)
                num4 = torch.tensor([-0.6179482974981999,1.2970445007662557,0.5731421077212038,1.1586797805331148,0.36384182727050746,-1.0,-0.2982976390813067]).to(device)
                num5 = torch.tensor([-0.7513287298392256,0.0017874858428588859,0.48213394107453317,1.1367623367923525,1.658043587840228,-1.0,-0.4137676929192318]).to(device)
                num6 = torch.tensor([0.89000431088073,0.6300831559505915,0.5438829989565102,-1.6322885741978679,-1.037565455320435,-1.0,0.2694301256218254]).to(device)
                num7 = torch.tensor([-0.14753098962267794,0.7465533357974641,0.43457706993274486,1.3130431316109905,-1.7256210615124672,-1.0,0.7409328454600198]).to(device)
                num8 = torch.tensor([-1.7231705795550696,0.8957063318532774,-0.9330216157034107,-1.4403467060512671,1.5167556976735066,-1.0,0.7120653320005386]).to(device)
                num9 = torch.tensor([0.6351478864898839,-0.3745802407432968,1.7243519955212836,-1.3580972204022161,1.587279308189756,-1.0,-0.22131760318935656]).to(device)
                labels = torch.stack([num1, num2, num3, num4, num5, num6, num7, num8, num9], dim=0).to(device)

                labels = labels.float()
                samples = model.sample(9, labels)
                save_samples(samples, labels, f"create/epoch{epoch+1}")
                
                # Add generated sample images to TensorBoard
                grid_samples = make_grid(samples, nrow=3)
                writer.add_image('Generated Images', grid_samples, epoch)
                
                # Save model checkpoint
                model_path = os.path.join("saved_models", f"model_epoch_{epoch+1}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': epoch_loss / num_batches if num_batches > 0 else float('inf'),
                    'args': vars(args)
                }, model_path)
                
                # Record model structure to TensorBoard (only the first time)
                if epoch + 1 == args.sample_after:
                    dummy_input = torch.randn(1, channels, image_size, image_size).to(device)
                    dummy_t = torch.tensor([0]).to(device).float() / time_steps
                    dummy_label = torch.zeros(1, label_dim).to(device)
                    try:
                        writer.add_graph(model.model, (dummy_input, dummy_t, dummy_label))
                    except Exception as e:
                        print(f"Failed to add model graph to TensorBoard: {e}")
    
    except KeyboardInterrupt:
        print(f"Training interrupted at epoch {current_epoch+1}")
        # Save model at interruption
        interrupted_model_path = os.path.join("saved_models", f"model_interrupted.pth")
        torch.save({
            'epoch': current_epoch,
            'model_state_dict': model.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'args': vars(args)
        }, interrupted_model_path)
        print(f"Model interrupted at {interrupted_model_path}")
        
        # Return current state so it can be handled in the main function
        return current_epoch, optimizer, scheduler
    
    writer.close()
    # Normal completion of training, return final state
    return num_epochs-1, optimizer, scheduler


def save_samples(samples, labels, base_filename):
    """
    Save generated sample images and corresponding parameters
    :param samples: Generated samples, shape is (n_samples, channels, height, width)
    :param labels: Labels for generated samples, each label contains 7 values
    :param base_filename: Prefix for saving file names
    """
    samples = samples.cpu().numpy()
    os.makedirs(base_filename, exist_ok=True)
    
    # Extract epoch number from base_filename
    epoch = int(base_filename.split('epoch')[-1])
    
    for i, label in enumerate(labels):
        # Process image data
        sample_T = np.transpose(samples[i], (1, 2, 0))
        # Scale image data to 0-255 range
        sample_T_scaled = (sample_T * 255).astype(np.uint8)
        img = Image.fromarray(sample_T_scaled.squeeze(), mode='L')  # 'L' 表示灰度模式
        
        # Generate file name
        label_str = '_'.join(f"{val:.8f}" for val in label)
        base_name = f"{base_filename}/{label_str}_epoch{epoch}"
        
        # Save image
        img_filename = f"{base_name}.png"
        img.save(img_filename)
        
        # Save npz file
        npz_filename = f"{base_name}.npz"
        np.savez(npz_filename,
                 image_array=sample_T,
                 parameters=label.cpu().numpy())


if __name__ == "__main__":
    dataloader, val_dataloader = load_data(args.val_split)
    diffusion_model = DiffusionModel()
    
    # Resume training checkpoint (if any)
    start_epoch = 0
    checkpoint = None
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Load checkpoint '{args.resume}'")
            try:
                checkpoint = torch.load(args.resume, map_location=device)
                
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    if args.distributed:
                        has_module_prefix = any(k.startswith('module.') for k in state_dict.keys())
                        
                        if not has_module_prefix:
                            print("Add 'module.' prefix to checkpoint state dictionary...")
                            state_dict = {f'module.{k}': v for k, v in state_dict.items()}
                    else:
                        has_module_prefix = any(k.startswith('module.') for k in state_dict.keys())
                        
                        if has_module_prefix:
                            print("Remove 'module.' prefix from checkpoint state dictionary...")
                            new_state_dict = {}
                            for k, v in state_dict.items():
                                if k.startswith('module.'):
                                    new_state_dict[k[7:]] = v 
                                else:
                                    new_state_dict[k] = v
                            state_dict = new_state_dict
                    
                    if args.distributed:
                        diffusion_model.model.load_state_dict(state_dict)
                    else:
                        diffusion_model.model.load_state_dict(state_dict)
                    
                    # Set starting epoch
                    if 'epoch' in checkpoint:
                        start_epoch = checkpoint['epoch'] + 1
                    
                    if args.is_master or not args.distributed:
                        print(f"Resume training from epoch {start_epoch}")
                        
                        # Print checkpoint information
                        print(f"Checkpoint information:")
                        if 'loss' in checkpoint:
                            print(f"  Loss: {checkpoint['loss']:.6f}")
                        if 'args' in checkpoint:
                            # Print key parameters
                            saved_args = checkpoint['args']
                            print(f"  Original learning rate: {saved_args.get('lr', 'N/A')}")
                            print(f"  Original learning rate scheduler: {saved_args.get('lr_scheduler', 'N/A')}")
                            print(f"  Original batch size: {saved_args.get('batch_size', 'N/A')}")
                            
                            # Check if current parameters match saved parameters
                            if saved_args.get('lr_scheduler') != args.lr_scheduler:
                                print(f"Warning: Current learning rate scheduler ({args.lr_scheduler}) does not match saved scheduler ({saved_args.get('lr_scheduler')})")
                            
                            if saved_args.get('batch_size') != args.batch_size:
                                print(f"Warning: Current batch size ({args.batch_size}) does not match saved batch size ({saved_args.get('batch_size')})")
                else:
                    print("Attempt to load model state dictionary directly...")
                    # Handle module prefix issue
                    has_module_prefix = any(k.startswith('module.') for k in checkpoint.keys())
                    
                    if args.distributed and not has_module_prefix:
                        print("Add 'module.' prefix to checkpoint state dictionary...")
                        new_checkpoint = {}
                        for k, v in checkpoint.items():
                            new_checkpoint[f'module.{k}'] = v
                        checkpoint = new_checkpoint
                    elif not args.distributed and has_module_prefix:
                        print("Remove 'module.' prefix from checkpoint state dictionary...")
                        new_checkpoint = {}
                        for k, v in checkpoint.items():
                            if k.startswith('module.'):
                                new_checkpoint[k[7:]] = v
                            else:
                                new_checkpoint[k] = v
                        checkpoint = new_checkpoint
                    
                    diffusion_model.model.load_state_dict(checkpoint)
                    
                    if args.is_master or not args.distributed:
                        print("Model weights loaded successfully")
            except Exception as e:
                if args.is_master or not args.distributed:
                    print(f"Error loading checkpoint: {e}")
                    import traceback
                    traceback.print_exc()
        else:
            if args.is_master or not args.distributed:
                print(f"Checkpoint '{args.resume}' not found")
    
    try:
        # Start training
        if args.is_master or not args.distributed:
            print(f"Start training - Device: {device}, Learning rate scheduler: {args.lr_scheduler}")
            
            # Print training configuration
            print(f"Training configuration:")
            print(f"  Total epochs: {num_epochs}")
            print(f"  Batch size: {batch_size}")
            print(f"  Learning rate: {args.lr}")
            print(f"  Loss weights - noise: {loss_noise_prop}, label: {loss_label_prop}")
            print(f"  Validation evaluation - every {args.val_every} epochs, using {args.val_samples if args.val_samples > 0 else 'all'} samples")
            
            if args.distributed:
                print(f"  Distributed training: using {dist.get_world_size()} processes")
        
        # Train model and get current state (for interruption recovery)
        current_epoch, optimizer, scheduler = train(diffusion_model, dataloader, num_epochs, start_epoch=start_epoch, val_dataloader=val_dataloader)
        
        # Save final model (only main process)
        if args.is_master or not args.distributed:
            model_path = os.path.join("saved_models", f"model_epoch_finished.pth")
            # Save module in distributed training
            model_state_dict = diffusion_model.model.module.state_dict() if args.distributed else diffusion_model.model.state_dict()
            torch.save({
                'epoch': current_epoch,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'args': vars(args)
            }, model_path)
            print(f"Training completed, model saved to {model_path}")
        
    except KeyboardInterrupt:
        if args.is_master or not args.distributed:
            print("Training interrupted in main function")
            interrupted_model_path = os.path.join("saved_models", f"model_interrupted.pth")
            # Save module in distributed training
            model_state_dict = diffusion_model.model.module.state_dict() if args.distributed else diffusion_model.model.state_dict()
            torch.save({
                'epoch': start_epoch-1 if 'start_epoch' in locals() else 0,
                'model_state_dict': model_state_dict,
                'args': vars(args)
            }, interrupted_model_path)
            print(f"Interrupted model saved to {interrupted_model_path}")
    except Exception as e:
        if args.is_master or not args.distributed:
            print(f"Training error: {e}")
            # Try to save current state
            if 'current_epoch' in locals() and current_epoch > start_epoch:
                error_model_path = os.path.join("saved_models", f"model_error.pth")
                # Save module in distributed training
                model_state_dict = diffusion_model.model.module.state_dict() if args.distributed else diffusion_model.model.state_dict()
                torch.save({
                    'epoch': current_epoch,
                    'model_state_dict': model_state_dict,
                    'optimizer_state_dict': optimizer.state_dict() if 'optimizer' in locals() else None,
                    'scheduler_state_dict': scheduler.state_dict() if 'scheduler' in locals() else None,
                    'error': str(e),
                    'args': vars(args)
                }, error_model_path)
                print(f"Error state saved to {error_model_path}")
            import traceback
            traceback.print_exc()
    
    # Add distributed training cleanup step
    if args.distributed:
        dist.destroy_process_group()