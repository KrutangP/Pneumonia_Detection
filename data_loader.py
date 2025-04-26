import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class PneumoniaDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = './chest_xray'  # Directory containing train/val/test
        self.transform = transform
        self.classes = ['NORMAL', 'PNEUMONIA']
        self.images = []
        self.labels = []

        # Iterate through train, val, or test directories
        for class_idx, class_name in enumerate(self.classes):
            class_folder = os.path.join(data_dir, class_name)
            for filename in os.listdir(class_folder):
                img_path = os.path.join(class_folder, filename)
                
                # Skip non-image files (like .DS_Store)
                if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                
                self.images.append(img_path)
                self.labels.append(class_idx)

    def __len__(self):
        return len(self.images)  # Number of images in the dataset

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        if self.transform:
            img = self.transform(img)
        return img, label


def load_data(batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize for grayscale images
    ])

    # Specify the directories for train and val datasets
    train_dir = 'chest_xray/train'
    val_dir = 'chest_xray/val'

    # Initialize the datasets for train and val
    train_dataset = PneumoniaDataset(train_dir, transform=transform)
    val_dataset = PneumoniaDataset(val_dir, transform=transform)

    # DataLoader setup for training and validation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def load_test_data(batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize for grayscale images
    ])

    # Specify the test directory
    test_dir = 'chest_xray/test'

    # Initialize the test dataset
    test_dataset = PneumoniaDataset(test_dir, transform=transform)

    # DataLoader setup for testing
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_loader
