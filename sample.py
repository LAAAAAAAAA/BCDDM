
# This file is part of BCDDM and is released under the BSD 3-Clause License.
# 
# Copyright (c) 2025 Zelin Zhang, Ao Liu. All rights reserved.

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import random
import time
import argparse
from tqdm import tqdm
import json
import torch.distributed as dist

from mpl_toolkits.axes_grid1 import make_axes_locatable


def parse_args():
    parser = argparse.ArgumentParser(description="BCDDM Sample script")
    parser.add_argument('--image_size', type=int, default=256, help='The size of the image (width and height)')
    parser.add_argument('--channels', type=int, default=1, help='The number of channels in the image')
    parser.add_argument('--time_steps', type=int, default=1000, help='The number of diffusion steps')
    parser.add_argument('--beta_start', type=float, default=0.0001, help='The starting value of β')
    parser.add_argument('--beta_end', type=float, default=0.02, help='The ending value of β')
    parser.add_argument('--label_dim', type=int, default=7, help='The dimension of the label')
    parser.add_argument('--seed', type=int, default=42, help='The random seed')
    
    parser.add_argument('--model_path', type=str, help='The path of the pre-trained model')
    parser.add_argument('--output_dir', type=str, default='samples', help='The output directory for generated images')
    parser.add_argument('--num_samples', type=int, default=9, help='The number of generated images')
    parser.add_argument('--batch_size', type=int, default=9, help='The number of samples per batch')
    
    parser.add_argument('--input_dir', type=str, default='', help='The path of the folder containing the original images and labels')
    parser.add_argument('--random_labels', action='store_true', help='Whether to generate random labels')
    parser.add_argument('--random_seed', type=int, default=42, help='The seed for random labels')
    parser.add_argument('--label_range', type=str, default='-1.7320508075688774,1.7320508075688774', help='The range of random labels, format: min,max')
    
    parser.add_argument('--config', type=str, default='', help='The path of the configuration file')
    
    parser.add_argument('--gpu', type=int, default=0, help='The GPU ID used')
    parser.add_argument('--distributed', action='store_true', help='Whether to use distributed environment')
    
    args = parser.parse_args()
    
    # If the configuration file is specified, load the parameters from the configuration file
    if args.config:
        if os.path.exists(args.config):
            with open(args.config, 'r') as f:
                config = json.load(f)
                
                if 'model' in config:
                    args.image_size = config['model'].get('image_size', args.image_size)
                    args.channels = config['model'].get('channels', args.channels)
                    args.time_steps = config['model'].get('time_steps', args.time_steps)
                    args.beta_start = config['model'].get('beta_start', args.beta_start)
                    args.beta_end = config['model'].get('beta_end', args.beta_end)
                    args.label_dim = config['model'].get('label_dim', args.label_dim)
                
                if 'sampling' in config:
                    args.model_path = config['sampling'].get('model_path', args.model_path)
                    args.output_dir = config['sampling'].get('output_dir', args.output_dir)
                    args.num_samples = config['sampling'].get('num_samples', args.num_samples)
                    args.batch_size = config['sampling'].get('batch_size', args.batch_size)
                
                if 'labels' in config:
                    args.input_dir = config['labels'].get('input_dir', args.input_dir)
                    args.random_labels = config['labels'].get('random', args.random_labels)
                    args.random_seed = config['labels'].get('seed', args.random_seed)
                    args.label_range = config['labels'].get('range', args.label_range)
                
                print(f"The configuration file '{args.config}' has been loaded")
        else:
            print(f"The configuration file '{args.config}' does not exist, using default values")
    
    return args

# Initialize the distributed training function
def init_distributed_mode(args):
    """
    Initialize the distributed environment
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
        args.is_master = (args.rank == 0)
        
        # Set the current device
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl')
        print(f'| Distributed initialization completed, number of processes: {dist.get_world_size()}')
    else:
        args.distributed = False
        args.is_master = True

# Set the random seed
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# Global parameter configuration
args = parse_args()

# Create the output directory
os.makedirs(args.output_dir, exist_ok=True)

# Initialize the distributed environment (if enabled)
if args.distributed:
    init_distributed_mode(args)
else:
    args.is_master = True

# Set the device
if args.distributed:
    device = torch.device(f"cuda:{args.local_rank}")
else:
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

# Set the random seed
seed_everything(args.seed)

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
# The dimension of the label
label_dim = args.label_dim

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
        :return: Feature map after double convolution
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
        
        self.time_embed = nn.Sequential(
            nn.Linear(1, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128)
        )

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
        # Time embedding
        t_emb = self.time_embed(t.unsqueeze(-1).float())
        
        # Label embedding
        label_emb = self.label_embed(label)
        
        # Combine condition information
        combined = t_emb + label_emb
        
        condition = combined.unsqueeze(-1).unsqueeze(-1)
        condition = condition.expand(-1, -1, x.shape[2], x.shape[3])
        
        # Concatenate input and condition
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
    def __init__(self, model_path):
        """
        Initialize the Diffusion Model
        :param model_path: The path of the pre-trained model
        """
        self.model = ConditionalUNet().to(device)
        self.load_model(model_path)
        
        self.betas = torch.linspace(beta_start, beta_end, time_steps).to(device)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculate the parameters required for pre-processing
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
    
    def load_model(self, model_path):
        """
        Load the pre-trained model
        :param model_path: The path of the pre-trained model
        """
        print(f"Loading the pre-trained model: {model_path}")
        try:
            checkpoint = torch.load(model_path, map_location=device)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                
                if any(k.startswith('module.') for k in state_dict.keys()):
                    print("Removing the 'module.' prefix from all keys...")
                    new_state_dict = {}
                    for k, v in state_dict.items():
                        if k.startswith('module.'):
                            new_state_dict[k[7:]] = v 
                        else:
                            new_state_dict[k] = v
                    state_dict = new_state_dict
                
                self.model.load_state_dict(state_dict)
                print("The model has been loaded successfully")
                
                # If the checkpoint contains parameter information, print it out
                if 'args' in checkpoint:
                    args_dict = checkpoint['args']
                    print(f"Model training parameters:")
                    for key in ['image_size', 'channels', 'time_steps', 'beta_start', 'beta_end', 'label_dim']:
                        if key in args_dict:
                            print(f"  {key}: {args_dict[key]}")
            else:
                # Try to load directly
                self.model.load_state_dict(checkpoint)
                print("The model has been loaded successfully")
        except Exception as e:
            print(f"Error loading the model: {e}")
            import traceback
            traceback.print_exc()
            exit(1)
        
        # Set the model to evaluation mode
        self.model.eval()

    def get_index_from_list(self, vals, t, x_shape):
        """
        Get the value of the specified time step from the alphas_cumprod list
        :param vals: The alphas_cumprod list, (T,) 
        :param t: The time step, (batch_size,)
        :param x_shape: The shape of the output
        :return: The value of the specified time step, the shape is the same as x_shape
        """
        batch_size = t.shape[0]
        out = vals.gather(-1, t)
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
        return out

    def forward_diffusion(self, x0, t):
        """
        Forward diffusion process
        :param x0: The original image, shape is (batch_size, channels, height, width)
        :param t: The time step, shape is (batch_size,)
        :return: The noisy image and noise, the shape is the same as x0
        """
        noise = torch.randn_like(x0)
        alphas_cumprod_t = self.get_index_from_list(self.alphas_cumprod, t, x0.shape)
        xt = torch.sqrt(alphas_cumprod_t) * x0 + torch.sqrt(1 - alphas_cumprod_t) * noise
        return xt, noise

    @torch.no_grad()
    def sample(self, n_samples, labels, size=None):
        """
        Sample and generate images
        :param n_samples: The number of generated images
        :param labels: The labels, shape is (n_samples, label_dim)
        :param size: The size of the generated image, default using image_size
        :return: The generated image, shape is (n_samples, channels, size, size)
        """
        if size is None:
            size = image_size
            
        print(f"Generate {n_samples} images...")
        x = torch.randn(n_samples, channels, size, size).to(device)

        for i in tqdm(reversed(range(time_steps)), desc="Sampling"):
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

        x = (x.clamp(-1, 1) + 1) / 2
        return x
            
            
    def train_step(self, x0, labels):
        """
        Training step - kept for use when necessary
        """
        pass

class ImageFileDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir: The path of the folder containing the images
        transform: The optional transformation applied to the images
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f)) and f.endswith('.npz')]
        print(f"Found {len(self.image_files)} npz files")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)

        try:
            data = np.load(img_path)
            
            image = np.array(data['I_rot'], dtype=np.float32)
            image = image / image.max()
            
            # Try to get the label data, based on possible key names
            if 'normalized_parm_array' in data:
                label = data['normalized_parm_array']
            else:
                # If the label is not found, return a default value or an empty array
                print(f"Warning: No label data found in file {img_name}")
                label = np.zeros(label_dim, dtype=np.float32)
            
            label = torch.tensor(label, dtype=torch.float32)
            
            # Apply the transformation (if any)
            if self.transform:
                image = self.transform(image)
            
            return image, label
            
        except Exception as e:
            print(f"Error loading file {img_path}: {e}")
            # Return a dummy image and label
            dummy_image = torch.zeros((1, image_size, image_size), dtype=torch.float32)
            dummy_label = torch.zeros(label_dim, dtype=torch.float32)
            return dummy_image, dummy_label

def restore_label(normalized_label):
    """
    Restore the label dimension
    """
    # parm_array = np.array([a, np.log10(Te_unit), disk_h, MBH, keplerian_factor, fluid_dirction, pa])
    sigma = np.sqrt(1/12)
    mean_parm_array = np.array([0, 11, 0.45, 6.5e9, 0.5, 0, 180])
    std_parm_array = np.array([2*sigma, 2*sigma, 0.7*sigma, 3e9*sigma, 1*sigma, 1, 360*sigma])

    parm_array = normalized_label * std_parm_array + mean_parm_array
    parm_array[1] = 10**parm_array[1]

    return parm_array

def make_plot(original_image_T, sample_T, save_dir, sample_idx, original_parm_array, pred_parm_array):
    """
    Generate a comparison plot of the image and label
    """
    # Restore the image dimension
    c_cgs = 2.99792458e10
    k_cgs = 1.38064852e-16
    freq = 230e9
    original_image_scale = 0.5/original_image_T.sum()
    original_image_T_cgs = original_image_T*original_image_scale
    sample_image_scale = 0.5/sample_T.sum()
    sample_image_T_cgs = sample_T*sample_image_scale

    original_image_Tb = c_cgs**2/(2*freq**2*k_cgs) * original_image_T_cgs
    sample_image_Tb = c_cgs**2/(2*freq**2*k_cgs) * sample_image_T_cgs

    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.4), dpi=300)
    
    # Display the original image
    im0 = axes[0].imshow(original_image_Tb, cmap='afmhot', vmin=0., vmax=original_image_Tb.max(), origin='upper')
    axes[0].axis('off')
    # Create a horizontal colorbar
    divider = make_axes_locatable(axes[0])
    cax = divider.append_axes("bottom", size="5%", pad=0.05)
    cbar0 = plt.colorbar(im0, cax=cax, orientation='horizontal')

    # Set the ticks, divide by 10e10 and only keep the integer positions
    vmax = original_image_Tb.max()
    if vmax/1e10 < 1:
        ticks = np.array([0, 0.5e10])
    else:
        ticks = np.arange(0, int(vmax/1e10)+1, 1)
        if len(ticks) > 6:
            ticks = ticks[::2]
        ticks = ticks * 1e10
    cbar0.set_ticks(ticks)
    if vmax/1e10 < 1:
        cbar0.set_ticklabels([f'{t/1e10:.1f}' for t in ticks])
    else:
        cbar0.set_ticklabels([f'{int(t/1e10)}' for t in ticks])

    # Display the generated image
    im1 = axes[1].imshow(sample_image_Tb, cmap='afmhot', vmin=0., vmax=sample_image_Tb.max(), origin='upper')
    axes[1].axis('off')
    # Create a horizontal colorbar
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes("bottom", size="5%", pad=0.05)
    cbar1 = plt.colorbar(im1, cax=cax, orientation='horizontal')

    # Set the ticks
    vmax = sample_image_Tb.max()
    if vmax/1e10 < 1:
        ticks = np.array([0, 0.5e10])
    else:
        ticks = np.arange(0, int(vmax/1e10)+1, 1)
        if len(ticks) > 6:
            ticks = ticks[::2]
        ticks = ticks * 1e10
    cbar1.set_ticks(ticks)
    if vmax/1e10 < 1:
        cbar1.set_ticklabels([f'{t/1e10:.1f}' for t in ticks])
    else:
        cbar1.set_ticklabels([f'{int(t/1e10)}' for t in ticks])

    # Display the original label and predicted label
    # Reorder the parameter names and corresponding indices
    param_names_ordered = ['M_{BH}', 'T_e', 'a', 'h', 'k', 'F_{dir}', 'PA']  # Parameter names (using LaTeX format)
    original_indices = [3, 1, 0, 2, 4, 5, 6]  # Index positions in the original array
    
    fontsize = 8
    x_pos = 0.02  
    y_start = 0.98  
    y_step = 0.1 

    for i, orig_idx in enumerate(original_indices):
        y_pos = y_start - (i * y_step)
        
        # Select different formatting methods based on the original index
        if orig_idx == 5:  # F_dir parameter
            orig_val = f"${param_names_ordered[i]}: {1 if original_parm_array[orig_idx] > 0 else -1}$"
            pred_val = f"${param_names_ordered[i]}: {1 if pred_parm_array[orig_idx] > 0 else -1}$"
        elif orig_idx in [1, 3]:  # Te and MBH use scientific notation
            orig_val = f"${param_names_ordered[i]}: {original_parm_array[orig_idx]:.2e}$"
            pred_val = f"${param_names_ordered[i]}: {pred_parm_array[orig_idx]:.2e}$"
        elif orig_idx == 6:  # PA parameter
            orig_val = f"${param_names_ordered[i]}: {original_parm_array[orig_idx]:.2f}$"
            pred_val = f"${param_names_ordered[i]}: {pred_parm_array[orig_idx]:.2f}$"
        else:  # Other parameters use normal format
            orig_val = f"${param_names_ordered[i]}: {original_parm_array[orig_idx]:.4f}$"
            pred_val = f"${param_names_ordered[i]}: {pred_parm_array[orig_idx]:.4f}$"
        
        # Original label
        axes[0].text(x_pos, y_pos, orig_val,
                    fontsize=fontsize,
                    color='white',
                    transform=axes[0].transAxes,
                    verticalalignment='top',
                    bbox=dict(facecolor='black', alpha=0.7, edgecolor='none', pad=1))
        
        # Predicted label
        axes[1].text(x_pos, y_pos, pred_val,
                    fontsize=fontsize,
                    color='white',
                    transform=axes[1].transAxes,
                    verticalalignment='top',
                    bbox=dict(facecolor='black', alpha=0.7, edgecolor='none', pad=1))

    # Save the image
    plt.savefig(os.path.join(save_dir, f'sample_{sample_idx}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    return original_image_Tb, sample_image_Tb

def make_plot_random(sample_T, save_dir, sample_idx, parm_array):
    # Restore the image dimension
    c_cgs = 2.99792458e10
    k_cgs = 1.38064852e-16
    freq = 230e9
    sample_image_scale = 0.5/sample_T.sum()
    sample_image_T_cgs = sample_T*sample_image_scale

    sample_image_Tb = c_cgs**2/(2*freq**2*k_cgs) * sample_image_T_cgs

    fig, axes = plt.subplots(1, 1, figsize=(4.8, 2.4), dpi=300)
    im = axes.imshow(sample_image_Tb, cmap='afmhot', vmin=0., vmax=sample_image_Tb.max(), origin='upper')
    axes.axis('off')
    divider = make_axes_locatable(axes)
    cax = divider.append_axes("bottom", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax, orientation='horizontal')

    # Set the ticks
    vmax = sample_image_Tb.max()
    if vmax/1e10 < 1:
        ticks = np.array([0, 0.5e10])
    else:
        ticks = np.arange(0, int(vmax/1e10)+1, 1)
        if len(ticks) > 6:
            ticks = ticks[::2]
        ticks = ticks * 1e10
    cbar.set_ticks(ticks)
    if vmax/1e10 < 1:
        cbar.set_ticklabels([f'{t/1e10:.1f}' for t in ticks])
    else:
        cbar.set_ticklabels([f'{int(t/1e10)}' for t in ticks])

    param_names_ordered = ['M_{BH}', 'T_e', 'a', 'h', 'k', 'F_{dir}', 'PA']  # Parameter names (using LaTeX format)
    original_indices = [3, 1, 0, 2, 4, 5, 6]  # Index positions in the original array
    
    fontsize = 8
    x_pos = 0.02  
    y_start = 0.98  
    y_step = 0.1 

    for i, orig_idx in enumerate(original_indices):
        y_pos = y_start - (i * y_step)
        
        # Select different formatting methods based on the original index
        if orig_idx == 5:  # F_dir parameter
            orig_val = f"${param_names_ordered[i]}: {1 if parm_array[orig_idx] > 0 else -1}$"
        elif orig_idx in [1, 3]:  # Te and MBH use scientific notation
            orig_val = f"${param_names_ordered[i]}: {parm_array[orig_idx]:.2e}$"
        elif orig_idx == 6:  # PA parameter
            orig_val = f"${param_names_ordered[i]}: {parm_array[orig_idx]:.2f}$"
        else:  # Other parameters use normal format
            orig_val = f"${param_names_ordered[i]}: {parm_array[orig_idx]:.4f}$"
        
        # Original label
        axes.text(x_pos, y_pos, orig_val,
                    fontsize=fontsize,
                    color='white',
                    transform=axes.transAxes,
                    verticalalignment='top',
                    bbox=dict(facecolor='black', alpha=0.7, edgecolor='none', pad=1))
    
    # Save the image
    plt.savefig(os.path.join(save_dir, f'sample_random_{sample_idx}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    return sample_image_Tb

# 主函数
if __name__ == "__main__":
    print("\n===== Generate BH images based on BCDDM =====")
    print(f"Model path: {args.model_path}")
    print(f"Output directory: {args.output_dir}")
    
    print(f"Image size: {image_size}x{image_size}, Channels: {channels}")
    print(f"Label dimension: {label_dim}")
    print(f"Random seed: {args.seed}")
    print("==============================\n")
    
    # Load the model
    diffusion_model = DiffusionModel(args.model_path)
    
    # Original images and labels
    original_images = None
    original_labels = None
    
    # Create the save directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    prefix = 'random' if args.random_labels else 'specified'
    save_dir = os.path.join(args.output_dir, f"{prefix}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Save the generated images to: {save_dir}")
    
    # Save the parameter information
    params_file = os.path.join(save_dir, "generation_params.json")
    generation_params = {
        "timestamp": timestamp,
        "num_samples": args.num_samples,
        "image_size": image_size,
        "channels": channels,
        "label_dim": label_dim,
    }
    
    
    with open(params_file, 'w') as f:
        json.dump(generation_params, f, indent=4)
    
    # If the input directory is provided, load the original images and labels
    if args.input_dir and os.path.exists(args.input_dir):
        print(f"Load the original images and labels from the input directory: {args.input_dir}")
        # Simple conversion, scale the image values to [-1, 1], consistent with the model input
        transform = transforms.Compose([
            transforms.ToTensor(),           # Convert to PyTorch tensor
            transforms.Normalize((0.5,), (0.5,))  # Normalize: Convert image values from [0,1] range to [-1,1] range, using mean and standard deviation (0.5,0.5)
        ])
        
        # Create the dataset and data loader
        dataset = ImageFileDataset(args.input_dir, transform=transform)
        
        # Limit the number of samples
        num_samples = min(args.num_samples, len(dataset))
        
        if num_samples < args.num_samples:
            print(f"警告：输入目录中的图像数量（{len(dataset)}）小于请求的数量（{args.num_samples}）")
            args.num_samples = num_samples
        
        # Use the data loader to load data
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        
        # Process data for each batch
        batch_idx = 0
        total_saved = 0
        
        print(f"\nStart generating images...")
        
        for batch_images, batch_labels in dataloader:
            # Calculate the size of the current batch
            batch_size = batch_images.shape[0]
            if total_saved + batch_size > args.num_samples:
                # If this batch will exceed the requested number of samples, only take the needed part
                batch_size = args.num_samples - total_saved
                batch_images = batch_images[:batch_size]
                batch_labels = batch_labels[:batch_size]
            
            # Move the data to the device
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)
            
            print(f"Generating the {batch_idx+1} batch ({batch_size} images)...")
            
            # Sample and generate images
            batch_samples = diffusion_model.sample(
                batch_size, 
                batch_labels
            )
            
            # Process the sampled images with the model, get the predicted labels
            with torch.no_grad():
                batch_t = torch.zeros(batch_size, dtype=torch.long, device=device)  # The time step is 0, indicating the final image
                _, predicted_batch_labels = diffusion_model.model(batch_images, batch_t / time_steps, batch_labels)
            
            # Immediately save the images generated in the current batch
            print(f"Saving the {batch_idx+1} batch generated images...")
            
            # Convert to NumPy arrays for saving
            samples_np = batch_samples.cpu().numpy()
            original_images_np = batch_images.cpu().numpy()
            labels_np = batch_labels.cpu().numpy()
            predicted_labels_np = predicted_batch_labels.cpu().numpy()
            
            # Save each sample
            for i in range(batch_size):
                sample_idx = total_saved + i
                
                # Process the generated image data
                sample_T = np.transpose(samples_np[i], (1, 2, 0))
                # sample_T = np.clip(sample_T, 0, 1)
                
                # Process the original image data
                original_image_T = np.transpose(original_images_np[i], (1, 2, 0))
                original_image_T = (original_image_T + 1) / 2
                
                # Get the labels
                label = labels_np[i]
                pred_label = predicted_labels_np[i]

                # Restore the label dimension
                original_parm_array = restore_label(label)
                pred_parm_array = restore_label(pred_label)

                # Save the generated image
                original_image_Tb, sample_image_Tb = make_plot(original_image_T, sample_T, save_dir, sample_idx, original_parm_array, pred_parm_array)
                
                # Save the npz file, containing the original image, generated image, original label and predicted label
                npz_filename = f"{save_dir}/data_{sample_idx:03d}.npz"
                np.savez(npz_filename,
                        original_image=original_image_T,
                        generated_image=sample_T,
                        original_label=label,
                        predicted_label=pred_label,
                        original_image_Tb=original_image_Tb,
                        generated_image_Tb=sample_image_Tb,
                        original_parm_array=original_parm_array,
                        predicted_parm_array=pred_parm_array)
            
            # Update the number of saved samples and batch index
            total_saved += batch_size
            batch_idx += 1
            
            print(f"Saved {total_saved}/{args.num_samples} images")
            
            # If the requested number of samples has been reached, exit the loop
            if total_saved >= args.num_samples:
                break
            
        print(f"\nSaved {total_saved} images")
    
    else:
        print("No input directory provided or the directory does not exist")
        print("Randomly generate labels")
        
        # parse the label range
        label_range = [float(x) for x in args.label_range.split(',')]
        
        # randomly generate labels
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        labels_np = np.random.uniform(label_range[0], label_range[1], (args.num_samples, args.label_dim))
        labels = torch.tensor(labels_np, dtype=torch.float32).to(device)
        
        # process the data
        batch_idx = 0
        total_saved = 0
        
        print(f"\nStart generating {args.num_samples} images...")
        
        # process the data in batches
        while total_saved < args.num_samples:
            # calculate the size of the current batch
            current_batch_size = min(args.batch_size, args.num_samples - total_saved)
            
            # get the labels of the current batch
            batch_labels = labels[batch_idx * args.batch_size:batch_idx * args.batch_size + current_batch_size]
            
            # generate images
            samples = diffusion_model.sample(current_batch_size, batch_labels)
            samples_np = samples.cpu().numpy()
            
            # save each sample
            for i in range(current_batch_size):
                sample_idx = total_saved + i
                
                # process the generated image data
                sample_T = np.transpose(samples_np[i], (1, 2, 0))
                
                # get the label
                label = labels_np[batch_idx * args.batch_size + i]
                
                # restore the label dimension
                original_parm_array = restore_label(label)
                
                # save the generated image
                sample_image_Tb = make_plot_random(sample_T, save_dir, sample_idx, original_parm_array)
                
                # save the npz file
                npz_filename = f"{save_dir}/data_{sample_idx:03d}.npz"
                np.savez(npz_filename,
                        generated_image=sample_T,
                        original_label=label,
                        generated_image_Tb=sample_image_Tb,
                        original_parm_array=original_parm_array)
            
            # update the number of saved samples and batch index
            total_saved += current_batch_size
            batch_idx += 1
            
            print(f"Saved {total_saved}/{args.num_samples} images")
    
    # Save all labels to a file
    if 'total_saved' in locals() and total_saved > 0:
        labels_file = os.path.join(save_dir, "all_labels.npz")
        
        if original_images is not None:
            np.savez(labels_file, 
                    original_labels=labels_np[:total_saved], 
                    predicted_labels=predicted_labels_np[:total_saved])
        else:
            np.savez(labels_file, 
                    original_labels=labels_np[:total_saved])
    
    print("\n===== Generation completed =====")
    print(f"The generated images have been saved to: {save_dir}")
    print("===============================")