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
    
    # Randomly sample safely
    sampled_ld = random.sample(ld_paths, min(target_ld_count, len(ld_paths)))
    sampled_sd = random.sample(sd_paths, min(target_sd_count, len(sd_paths)))
    
    # Labels: 0 for Real, 1 for Fake
    dataset_paths = [(p, 0) for p in real_paths] + [(p, 1) for p in sampled_ld] + [(p, 1) for p in sampled_sd]
    random.shuffle(dataset_paths)
    
    return dataset_paths

class ArtDataset(Dataset):
    """
    Custom PyTorch Dataset that loads images from a list of paths.
    """
    def __init__(self, file_paths_labels, transform=None):
        self.file_paths_labels = file_paths_labels
        self.transform = transform
        
    def __len__(self):
        return len(self.file_paths_labels)
        
    def __getitem__(self, idx):
        img_path, label = self.file_paths_labels[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            # Fallback for corrupted images if any exist in dataset
            placeholder = torch.zeros(3, 224, 224)
            return placeholder, label

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

def main():
    # --- CONFIGURATION HYPERPARAMETERS ---
    BATCH_SIZE = 256          # Increased to maximize RTX 5070 performance
    NUM_WORKERS = 8         # Multi-threaded background data loading
    MAX_TRAIN_SAMPLES = 10000 # Subsample limit for faster execution right now
    MAX_TEST_SAMPLES = 2000   # Subsample limit for testing
    NUM_EPOCHS = 10
    # -------------------------------------

    print("Downloading/Locating dataset from Kaggle...")
    dataset_path = kagglehub.dataset_download("ravidussilva/real-ai-art")
    print(f"Dataset path: {dataset_path}")
    
    train_dir = None
    test_dir = None
    for root, dirs, files in os.walk(dataset_path):
        if 'train' in dirs and train_dir is None:
            train_dir = os.path.join(root, 'train')
        if 'test' in dirs and test_dir is None:
            test_dir = os.path.join(root, 'test')
            
    if not train_dir or not test_dir:
        print("Could not find 'train' or 'test' directories.")
        return

    print("\nPreparing balanced dataset paths...")
    train_paths = get_balanced_dataset_paths(train_dir)
    test_paths = get_balanced_dataset_paths(test_dir)
    
    # Apply subsampling limits for speed
    train_paths = train_paths[:MAX_TRAIN_SAMPLES]
    test_paths = test_paths[:MAX_TEST_SAMPLES]
    
    print(f"Subsampled Train size: {len(train_paths)}")
    print(f"Subsampled Test size:  {len(test_paths)}")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = ArtDataset(train_paths, transform=transform)
    test_dataset = ArtDataset(test_paths, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    history = {'train_acc': [], 'test_acc': [], 'train_loss': []}
    
    print("\nStarting training...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        total_batches = len(train_loader)
        for batch_idx, (inputs, labels) in enumerate(train_loader):
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
            
            # Print batch progress every 10 steps
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                print(f"  Epoch [{epoch+1}/{NUM_EPOCHS}] | Batch [{batch_idx+1}/{total_batches}] Processing...")
            
        epoch_loss = running_loss / len(train_dataset)
        train_acc = correct_train / total_train * 100
        
        # Validation Pass
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
        
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        
        print(f">> Epoch [{epoch+1}/{NUM_EPOCHS}] Finished - Loss: {epoch_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%\n")
        
    print("Training finished. Saving history plot...")
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, NUM_EPOCHS + 1), history['train_acc'], label='Train Accuracy', color='blue')
    plt.plot(range(1, NUM_EPOCHS + 1), history['test_acc'], label='Test Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs Epochs')
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_history.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {plot_path}")

if __name__ == "__main__":
    main()