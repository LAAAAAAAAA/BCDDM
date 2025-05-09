
# This file is part of BCDDM and is released under the BSD 3-Clause License.
# 
# Copyright (c) 2025 Ao Liu. All rights reserved.


import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
import logging
import random

# Configuration parameters
BATCH_SIZE = 256
SEED = 42
NUM_EPOCHS = 500
NUM_SAVR = 10
BASE_LR = 3e-3
GAMMA = 0.97
NUM_SCHEDULER = 50
PATIENCE = 10
MIN_LR = 1e-6
DATA_PATH = "ResNet_input"

PARAM_NAMES = {
    0: "a",
    1: "T_e",
    2: "h_disk",
    3: "M_BH",
    4: "k",
    5: "F_dir",
    6: "PA", 
}

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(SEED)

class BlackHoleImageDataset(Dataset):
    """Dataset loader for black hole simulation data"""
    
    def __init__(self, root_dir, transform=None, param_idx=0):
        """
        Args:
            root_dir (string): Directory with simulation files
            transform (callable, optional): Optional transforms
        """
        self.root_dir = root_dir
        self.transform = transform
        self.param_idx = param_idx
        self.file_list = [f for f in os.listdir(root_dir) 
                         if os.path.isfile(os.path.join(root_dir, f))]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.root_dir, self.file_list[idx])
        data = np.load(file_path)
        
        # Process image data
        image = np.array(data['I_rot'], dtype=np.float32)
        image /= image.max()
        
        # Process parameters
        params = data['normalized_parm_array'][self.param_idx]
        label = torch.tensor(params, dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def create_data_loader(data_dir, param_idx):
    """Create data loader with preprocessing pipeline"""
    transform = transforms.Compose([
        transforms.Lambda(lambda x: torch.from_numpy(x).float()),
        transforms.Lambda(lambda x: x.reshape(1, 256, 256) if x.dim() == 2 else 
                         x.squeeze(-1).reshape(1, 256, 256) if x.shape[-1] == 1 else x),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    dataset = BlackHoleImageDataset(root_dir=data_dir, transform=transform, param_idx=param_idx)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

class ParameterRegressionResNet(models.ResNet):
    """Custom ResNet model for parameter regression"""
    
    def __init__(self):
        super().__init__(block=models.resnet.BasicBlock, 
                        layers=[3, 4, 6, 3], 
                        num_classes=1)
        # Modify first convolution layer for single-channel input
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, 
                              stride=2, padding=3, bias=False)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.fc.in_features, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

def calculate_r2(true_values, pred_values):
    """Calculate R² score"""
    mean_true = np.mean(true_values)
    ss_total = np.sum((true_values - mean_true)**2)
    ss_residual = np.sum((true_values - pred_values)**2)
    return 1 - (ss_residual / ss_total)

def validate(model, val_loader, device, param_idx):
    """Model validation routine"""
    model.eval()
    r2_scores = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device).float()
            
            outputs = model(images)
            
            # Special processing for parameter 5
            if param_idx == 5:
                outputs = (outputs > 0).float() * 1. + (outputs <= 0).float() * -1.
                
            # Calculate metrics
            labels_np = labels.cpu().numpy().flatten()
            outputs_np = outputs.cpu().numpy().flatten()
            current_r2 = calculate_r2(labels_np, outputs_np)
            r2_scores.append(current_r2)
                
    return np.mean(r2_scores)

def train_model(model, train_loader, val_loader, device, param_idx):
    """Model training routine"""
    # Initialize training components
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=BASE_LR)
    scheduler = StepLR(optimizer, step_size=5, gamma=GAMMA)
    LOSS_HISTORY = []
    R2_HISTORY = []
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'ResNet_results/train/training_log.log'),
            logging.StreamHandler()
        ]
    )
    
    # Training loop
    best_val_r2 = -np.inf
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        
        for images, labels in train_loader:
            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels.unsqueeze(1))
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * images.size(0)
            
        # Update learning rate
        avg_loss = epoch_loss / len(train_loader.dataset)
        LOSS_HISTORY.append(avg_loss)
        
        if epoch > NUM_SCHEDULER and len(LOSS_HISTORY) > PATIENCE:
            if avg_loss > min(LOSS_HISTORY[-PATIENCE:]) and scheduler.get_last_lr()[0] > MIN_LR:
                scheduler.step()
                
        # Validation and logging
        logging.info(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {avg_loss:.4f} - LR: {scheduler.get_last_lr()[0]:.2e}")
        
        if (epoch+1) % NUM_SAVR == 0:
            val_r2 = validate(model, val_loader, device, param_idx)
            R2_HISTORY.append(val_r2)
            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
            model_path = f"ResNet_results/saved_models/param_{PARAM_NAMES[param_idx]}.pth"
            torch.save(model.state_dict(), model_path)
                
    # Save training artifacts
    save_training_results(param_idx, LOSS_HISTORY, R2_HISTORY, best_val_r2)

def save_training_results(param_idx, loss_history, r2_history, best_val_r2):
    """Save training plots and metrics"""
    plt.figure(figsize=(12, 6))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(loss_history, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # R2s plot
    plt.subplot(1, 2, 2)
    plt.plot(r2_history, label='Training R2')
    plt.xlabel('Epochs')
    plt.ylabel('R2')
    plt.legend()
    
    # Save plots
    plt.savefig(f'ResNet_results/train/training_plots_param_{PARAM_NAMES[param_idx]}.png')
    plt.close()
    
    # Save metrics
    with open(f'ResNet_results/train/training_metrics_param_{PARAM_NAMES[param_idx]}.txt', 'w') as f:
        f.write(f"Best Validation R2: {best_val_r2}\n")
        for epoch, loss in enumerate(loss_history):
            f.write(f"Epoch {epoch+1}: {loss}\n")

def main():
    """Main execution routine"""
    # Create output directories
    os.makedirs('ResNet_results/saved_models', exist_ok=True)
    os.makedirs('ResNet_results/train', exist_ok=True)
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Train for each parameter
    for param_idx in PARAM_NAMES.keys():
        print(f"\n=== Training for Parameter {PARAM_NAMES[param_idx]} ===")
        
        # Initialize model
        model = ParameterRegressionResNet().to(device)
        
        # Create data loaders
        train_loader = create_data_loader(f"{DATA_PATH}/train", param_idx)
        val_loader = create_data_loader(f"{DATA_PATH}/val", param_idx)
        
        # Train model
        train_model(model, train_loader, val_loader, device, param_idx)

if __name__ == "__main__":
    main()