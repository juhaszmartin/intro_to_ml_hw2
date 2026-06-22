import os
import sys
import random
import logging
import kagglehub
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn.functional as F
from PIL import Image
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# =====================================================================
# 1. LOGGING & INITIAL CONFIGURATION
# =====================================================================
def setup_logging(log_filename="training_execution.log"):
    """Configures logging to output to both console and a log file."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_filename, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )

# =====================================================================
# 2. DATA UTILITIES & BALANCED SEPARATION
# =====================================================================
def get_exact_balanced_paths(split_dir, max_total_samples):
    """
    Returns an EXACTLY balanced list of paths based on max_total_samples.
    Target: 50% Real, 25% LD, 25% SD
    """
    real_paths, ld_paths, sd_paths = [], [], []
    
    for root, dirs, files in os.walk(split_dir):
        img_files = [os.path.join(root, f) for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not img_files:
            continue
            
        category = os.path.basename(root)
        if category.startswith('AI_LD_'):
            ld_paths.extend(img_files)
        elif category.startswith('AI_SD_'):
            sd_paths.extend(img_files)
        else:
            real_paths.extend(img_files)
            
    # Calculate exact quotas
    target_real = max_total_samples // 2
    target_ld = max_total_samples // 4
    target_sd = max_total_samples - target_real - target_ld 
    
    sampled_real = random.sample(real_paths, min(target_real, len(real_paths)))
    sampled_ld = random.sample(ld_paths, min(target_ld, len(ld_paths)))
    sampled_sd = random.sample(sd_paths, min(target_sd, len(sd_paths)))
    
    dataset_paths = (
        [(p, 0, 'real') for p in sampled_real] + 
        [(p, 1, 'ld') for p in sampled_ld] + 
        [(p, 1, 'sd') for p in sampled_sd]
    )
    random.shuffle(dataset_paths)
    return dataset_paths

class ArtDataset(Dataset):
    def __init__(self, file_paths_meta, transform=None):
        self.file_paths_meta = file_paths_meta
        self.transform = transform
        
    def __len__(self):
        return len(self.file_paths_meta)
        
    def __getitem__(self, idx):
        img_path, label, source_type = self.file_paths_meta[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label, source_type
        except Exception as e:
            return torch.zeros(3, 224, 224), label, source_type

# --- CUSTOM AUGMENTATION ---
class AddGaussianNoise(object):
    """Adds subtle Gaussian noise to a tensor to act as a regularizer."""
    def __init__(self, mean=0., std=0.02):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise

# =====================================================================
# 3. MODEL ARCHITECTURES
# =====================================================================

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class DeeperCustomCNN(nn.Module):
    def __init__(self):
        super(DeeperCustomCNN, self).__init__()
        self.in_channels = 32
        self.init_block = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.layer1 = ResidualBlock(32, 32, stride=1)
        self.layer2 = ResidualBlock(32, 64, stride=2)
        self.layer3 = ResidualBlock(64, 128, stride=2)
        self.layer4 = ResidualBlock(128, 256, stride=2)
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 2)
        )
        
    def forward(self, x):
        x = self.init_block(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x

def get_resnet50_transfer_model():
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 128),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(128, 2)
    )
    return model

# =====================================================================
# 4. GRAD-CAM UTILITY FOR EXPLAINABILITY
# =====================================================================
class MinimalGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.forward_hook = target_layer.register_forward_hook(self._save_activation)
        self.backward_hook = target_layer.register_full_backward_hook(self._save_gradient)
        
    def _save_activation(self, module, input, output):
        self.activations = output.detach()
        
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
        
    def generate_heatmap(self, input_tensor, target_class):
        self.model.eval()
        
        # Unfreeze temporarily for frozen layers (like ResNet)
        original_states = []
        for param in self.model.parameters():
            original_states.append(param.requires_grad)
            param.requires_grad = True
            
        input_tensor.requires_grad_(True)
        output = self.model(input_tensor)
        self.model.zero_grad()
        loss = output[0, target_class]
        loss.backward()
        
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = F.interpolate(cam, size=(input_tensor.shape[2], input_tensor.shape[3]), mode='bilinear', align_corners=False)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-7)
        
        # Restore states
        for param, state in zip(self.model.parameters(), original_states):
            param.requires_grad = state
            
        return cam.squeeze().detach().cpu().numpy()
        
    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()

def save_gradcam_visualization(model, target_layer, sample_img_tensor, output_path, model_name, target_class=1):
    try:
        grad_cam = MinimalGradCAM(model, target_layer)
        input_batch = sample_img_tensor.unsqueeze(0)
        heatmap = grad_cam.generate_heatmap(input_batch, target_class)
        grad_cam.remove_hooks()
        
        img_np = sample_img_tensor.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
        img_np = np.clip(img_np, 0, 1)
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(img_np)
        axes[0].set_title("Original Transformed Image")
        axes[0].axis('off')
        
        axes[1].imshow(img_np)
        axes[1].imshow(heatmap, cmap='jet', alpha=0.5)
        axes[1].set_title(f"Grad-CAM Heatmap")
        axes[1].axis('off')
        
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        logging.info(f"Successfully saved Grad-CAM plot to: {output_path}")
    except Exception as e:
        logging.error(f"Failed generating Grad-CAM visualization for {model_name}: {e}")

# =====================================================================
# 5. MODULAR MODEL TRAINING LOOP
# =====================================================================
def train_and_evaluate(model, model_name, train_loader, test_loader, num_epochs, device, target_layer_for_cam=None):
    logging.info(f"\n========================================\nSTARTING TRAINING: {model_name}\n========================================")
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)    
    
    history = {
        'train_loss': [], 'train_acc': [], 'test_acc_overall': [],
        'test_acc_real': [], 'test_acc_ld': [], 'test_acc_sd': []
    }
    
    for epoch in range(num_epochs):
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0
        
        for inputs, labels, _ in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / len(train_loader.dataset)
        train_acc = (correct_train / total_train) * 100
        
        model.eval()
        all_preds, all_labels, all_sources = [], [], []
        validation_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels, sources in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                validation_loss += loss.item() * inputs.size(0)
                
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_sources.extend(sources)
                
        epoch_val_loss = validation_loss / len(test_loader.dataset)
        scheduler.step(epoch_val_loss)
        
        preds_np, labels_np, sources_np = np.array(all_preds), np.array(all_labels), np.array(all_sources)
        mask_real, mask_ld, mask_sd = (sources_np == 'real'), (sources_np == 'ld'), (sources_np == 'sd')
        
        acc_overall = (preds_np == labels_np).mean() * 100 if len(labels_np) > 0 else 0.0
        acc_real = (preds_np[mask_real] == labels_np[mask_real]).mean() * 100 if mask_real.sum() > 0 else 0.0
        acc_ld = (preds_np[mask_ld] == labels_np[mask_ld]).mean() * 100 if mask_ld.sum() > 0 else 0.0
        acc_sd = (preds_np[mask_sd] == labels_np[mask_sd]).mean() * 100 if mask_sd.sum() > 0 else 0.0
        
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(train_acc)
        history['test_acc_overall'].append(acc_overall)
        history['test_acc_real'].append(acc_real)
        history['test_acc_ld'].append(acc_ld)
        history['test_acc_sd'].append(acc_sd)
        
        logging.info(
            f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f} | "
            f"Train Acc: {train_acc:.1f}% | Test Acc: {acc_overall:.1f}% "
            f"(Real: {acc_real:.1f}%, LD: {acc_ld:.1f}%, SD: {acc_sd:.1f}%)"
        )
        
    safe_name = model_name.lower().replace(' ', '_')
    
    # Save Plots
    plt.figure(figsize=(10, 5))
    epochs_range = range(1, num_epochs + 1)
    plt.plot(epochs_range, history['train_acc'], label='Train Accuracy', linestyle='--')
    plt.plot(epochs_range, history['test_acc_overall'], label='Test Accuracy (Overall)', linewidth=2)
    plt.plot(epochs_range, history['test_acc_real'], label='Test Accuracy (Real)')
    plt.plot(epochs_range, history['test_acc_ld'], label='Test Accuracy (LD)')
    plt.plot(epochs_range, history['test_acc_sd'], label='Test Accuracy (SD)')
    plt.title(f'{model_name} Training History')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{safe_name}_history.png", dpi=200, bbox_inches='tight')
    plt.close()
    
    # Save Confusion Matrix
    cm = confusion_matrix(labels_np, preds_np)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake/AI'], yticklabels=['Real', 'Fake/AI'])
    plt.ylabel('Ground Truth Label')
    plt.xlabel('Predicted Label')
    plt.title(f'{model_name} Confusion Matrix')
    plt.savefig(f"{safe_name}_confusion_matrix.png", dpi=200, bbox_inches='tight')
    plt.close()
    
    if target_layer_for_cam is not None:
        for inputs, _, sources in test_loader:
            fake_indices = [i for i, s in enumerate(sources) if s in ['ld', 'sd']]
            if fake_indices:
                target_idx = fake_indices[0]
                sample_img = inputs[target_idx].to(device)
                save_gradcam_visualization(model, target_layer_for_cam, sample_img, f"{safe_name}_gradcam.png", model_name, target_class=1)
                break
                
    logging.info(f"Finished evaluation execution for {model_name}.\n")
    return history

# =====================================================================
# 6. MAIN ORCHESTRATION
# =====================================================================
def main():
    setup_logging()
    
    BATCH_SIZE = 128
    NUM_WORKERS = 4
    MAX_TRAIN_SAMPLES = 8000
    MAX_TEST_SAMPLES = 2000
    NUM_EPOCHS = 20 # Increased to 20 per request
    
    logging.info("Downloading/Locating dataset from Kaggle...")
    dataset_path = kagglehub.dataset_download("ravidussilva/real-ai-art")
    
    train_dir, test_dir = None, None
    for root, dirs, _ in os.walk(dataset_path):
        if 'train' in dirs and train_dir is None: train_dir = os.path.join(root, 'train')
        if 'test' in dirs and test_dir is None: test_dir = os.path.join(root, 'test')

    logging.info("Assembling exact 50/50 splits...")
    train_paths = get_exact_balanced_paths(train_dir, MAX_TRAIN_SAMPLES)
    test_paths = get_exact_balanced_paths(test_dir, MAX_TEST_SAMPLES)
    
    # --- TRANSFORMS ---
    # 1. Clean Base Transform (Test & Clean Train)
    base_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 2. Augmented Transform (Flips, Rotation, Mild Gaussian Noise)
    aug_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        AddGaussianNoise(mean=0.0, std=0.02),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # --- DATASETS & LOADERS ---
    train_dataset_clean = ArtDataset(train_paths, transform=base_transform)
    train_dataset_aug = ArtDataset(train_paths, transform=aug_transform)
    test_dataset = ArtDataset(test_paths, transform=base_transform)
    
    train_loader_clean = DataLoader(train_dataset_clean, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    train_loader_aug = DataLoader(train_dataset_aug, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # ---------------- 5 EXPLICIT RUNS ----------------

    # Run 1: SimpleCNN Clean
    model_simple_clean = SimpleCNN()
    train_and_evaluate(model_simple_clean, "SimpleCNN Clean", train_loader_clean, test_loader, NUM_EPOCHS, device, model_simple_clean.features[-3])
    
    # Run 2: SimpleCNN Augmented
    model_simple_aug = SimpleCNN()
    train_and_evaluate(model_simple_aug, "SimpleCNN Augmented", train_loader_aug, test_loader, NUM_EPOCHS, device, model_simple_aug.features[-3])

    # Run 3: DeeperCNN Clean
    model_deeper_clean = DeeperCustomCNN()
    train_and_evaluate(model_deeper_clean, "DeeperCNN Clean", train_loader_clean, test_loader, NUM_EPOCHS, device, model_deeper_clean.layer4.conv2)
    
    # Run 4: DeeperCNN Augmented
    model_deeper_aug = DeeperCustomCNN()
    train_and_evaluate(model_deeper_aug, "DeeperCNN Augmented", train_loader_aug, test_loader, NUM_EPOCHS, device, model_deeper_aug.layer4.conv2)

    # Run 5: ResNet50 Transfer Learning (Clean)
    model_resnet = get_resnet50_transfer_model()
    train_and_evaluate(model_resnet, "ResNet50 Transfer Learning", train_loader_clean, test_loader, NUM_EPOCHS, device, model_resnet.layer4[-1].conv3)

    logging.info("\nAll 5 Training Runs Complete!")

if __name__ == "__main__":
    main()