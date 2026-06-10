import os
import random
import kagglehub
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

def get_balanced_dataset_paths(split_dir):
    """
    Scans the split directory and returns a balanced list of (image_path, label).
    Real images are label 0, Fake images are label 1.
    Keeps all Real images, and takes an equal number of Fake images (50% LD, 50% SD).
    """
    real_paths = []
    ld_paths = []
    sd_paths = []
    
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
            
    # Subsample AI images to match Real images count (50/50 split)
    target_fake_count = len(real_paths)
    target_ld_count = target_fake_count // 2
    target_sd_count = target_fake_count - target_ld_count
    
    # Randomly sample
    sampled_ld = random.sample(ld_paths, min(target_ld_count, len(ld_paths)))
    sampled_sd = random.sample(sd_paths, min(target_sd_count, len(sd_paths)))
    
    # Labels: 0 for Real, 1 for Fake
    dataset_paths = [(p, 0) for p in real_paths] + [(p, 1) for p in sampled_ld] + [(p, 1) for p in sampled_sd]
    random.shuffle(dataset_paths)
    
    return dataset_paths

class ArtDataset(Dataset):
    """
    Custom PyTorch Dataset that loads images from a list of paths.
    This avoids having to physically move/copy 120,000 files on your disk.
    """
    def __init__(self, file_paths_labels, transform=None):
        self.file_paths_labels = file_paths_labels
        self.transform = transform
        
    def __len__(self):
        return len(self.file_paths_labels)
        
    def __getitem__(self, idx):
        img_path, label = self.file_paths_labels[idx]
        # Ensure image is RGB (some might be grayscale)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

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
            nn.Linear(128, 2) # 2 classes: 0 (Real) vs 1 (Fake)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def main():
    print("Downloading/Locating dataset from Kaggle...")
    # This downloads the dataset to cache if it doesn't exist, or returns the cached path
    dataset_path = kagglehub.dataset_download("ravidussilva/real-ai-art")
    print(f"Dataset path: {dataset_path}")
    
    # Locate train and test folders in the downloaded dataset
    train_dir = None
    test_dir = None
    
    for root, dirs, files in os.walk(dataset_path):
        if 'train' in dirs and train_dir is None:
            train_dir = os.path.join(root, 'train')
        if 'test' in dirs and test_dir is None:
            test_dir = os.path.join(root, 'test')
            
    if not train_dir or not test_dir:
        print("Could not find 'train' or 'test' directories in the dataset.")
        return

    print("\nPreparing balanced dataset paths (50% Real, 50% Fake)...")
    train_paths = get_balanced_dataset_paths(train_dir)
    test_paths = get_balanced_dataset_paths(test_dir)
    
    print(f"Train size: {len(train_paths)} (Real vs Fake: 50/50)")
    print(f"Test size:  {len(test_paths)} (Real vs Fake: 50/50)")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = ArtDataset(train_paths, transform=transform)
    test_dataset = ArtDataset(test_paths, transform=transform)
    
    # Set num_workers=0 to prevent multiprocessing issues on Windows when running scripts
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 100
    # Store history here
    history = {'train_acc': [], 'test_acc': [], 'train_loss': []}
    
    print("\nStarting training...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in train_loader:
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
            
        epoch_loss = running_loss / len(train_dataset)
        train_acc = correct_train / total_train * 100
        
        # Validation pass
        model.eval()
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()
                
        test_acc = correct_test / total_test * 100
        
        # Save metrics to history
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")
        
    print("\nTraining finished. Saving history plot...")
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), history['train_acc'], label='Train Accuracy', color='blue')
    plt.plot(range(1, num_epochs + 1), history['test_acc'], label='Test Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs Epochs')
    plt.legend()
    plt.grid(True)
    
    # Save instead of show
    plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_history.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    print(f"Saved plot to {plot_path}")

if __name__ == "__main__":
    main()
