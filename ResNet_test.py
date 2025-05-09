
# This file is part of BCDDM and is released under the BSD 3-Clause License.
# 
# Copyright (c) 2025 Ao Liu. All rights reserved.


import os
import logging
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms, models
from PIL import Image

# Configuration parameters
BATCH_SIZE = 256
SEED = 42
NUM_EPOCHS = 500
LEARNING_RATE = 3e-3
GAMMA = 0.95
DATA_PATH = "GRRT_gauss0"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PARAM_NAMES = {
    0: "a",
    1: "T_e",
    2: "h_disk",
    3: "M_BH",
    4: "k",
    5: "F_dir",
    6: "PA",
}

def seed_everything(seed):
    """Set random seed for reproducibility across multiple libraries"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(SEED)

class ImageFileDataset(Dataset):
    """Custom dataset for loading numpy files with image and parameter data"""
    
    def __init__(self, root_dir, transform=None, param_idx=0):
        """
        Args:
            root_dir (str): Directory containing the .npz files
            transform (callable, optional): Optional transform to be applied
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
        
        # Process image
        image = data['I_rot'].astype(np.float32)
        image = image / image.max()  # Normalize to [0, 1]
        
        # Process label
        label = data['normalized_parm_array'][self.param_idx]
        label = torch.tensor(label, dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def create_test_loader(data_dir, param_idx):
    """Create test data loader for specific parameter"""
    transform = transforms.Compose([
        transforms.Lambda(lambda x: torch.from_numpy(x).float()),
        transforms.Lambda(lambda x: x.reshape(1, 256, 256) if x.dim() == 2 else 
                        x.squeeze(-1).reshape(1, 256, 256) if x.shape[-1] == 1 else x),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    test_dataset = ImageFileDataset(root_dir=data_dir, transform=transform, param_idx=param_idx)
    return DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

class RegressionResNet(models.ResNet):
    """Custom ResNet variant for regression tasks with single-channel input"""
    
    def __init__(self):
        super().__init__(block=models.resnet.BasicBlock, 
                        layers=[3, 4, 6, 3], 
                        num_classes=1)
        # Modify first convolution for single-channel input
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, 
                              padding=3, bias=False)
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

def calculate_r2(y_true, y_pred):
    """Calculate R-squared coefficient"""
    # Ensure inputs are NumPy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_mean = np.mean(y_true)

    total_sum_sq = np.sum((y_true - y_mean) ** 2)
    resid_sum_sq = np.sum((y_true - y_pred) ** 2)
    return 1 - (resid_sum_sq / total_sum_sq)

def evaluate_model(test_loader, model, parameter_idx):
    """Evaluate model performance on test set"""
    model.eval()
    predictions = []
    labels = []
    
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            
            # Special processing for F_dir parameter
            if parameter_idx == 5:
                outputs = (outputs > 0).float() * 1. + (outputs <= 0).float() * -1.
            
            predictions.extend(outputs.cpu().numpy().flatten())
            labels.extend(targets.unsqueeze(1).cpu().numpy().flatten())
    
    # Calculate metrics
    r2 = calculate_r2(labels, predictions)
    print(f'R² Score: {r2:.6f}')
    
    # Save results
    os.makedirs('ResNet_results/test', exist_ok=True)
    result_path = f'ResNet_results/test/{PARAM_NAMES[parameter_idx]}_{r2:.6f}.txt'
    
    with open(result_path, 'w') as f:
        f.write("Sample Index,Prediction,Label\n")
        for idx, (pred, lbl) in enumerate(zip(predictions, labels)):
            f.write(f"{idx},{pred:.6f},{lbl:.6f}\n")
            
    print(f"Results saved to: {result_path}")
    return r2

def main():
    """Main execution function"""
    # Create output directories
    os.makedirs('ResNet_results/test', exist_ok=True)
    
    
    # Evaluate for all parameters
    r2_scores = []
    for param_idx in PARAM_NAMES.keys():
        print(f"\n=== Testing parameter {PARAM_NAMES[param_idx]} ===")
        
        # Initialize components
        test_loader = create_test_loader(f"{DATA_PATH}/test", param_idx)
        model = RegressionResNet().to(DEVICE)
        
        # Load trained weights
        model_path = f'ResNet_results/saved_models/param_{PARAM_NAMES[param_idx]}.pth'
        model.load_state_dict(torch.load(model_path))
        
        # Evaluate
        score = evaluate_model(test_loader, model, param_idx)
        r2_scores.append(score)
    
    # Print final summary
    print("\n=== Final Results ===")
    for idx, score in enumerate(r2_scores):
        print(f"{PARAM_NAMES[idx]}: {score:.4f}")

if __name__ == "__main__":
    main()