"""
Dataset loaders for Adult Income, CIFAR-10, and PatchCamelyon (PCam)
"""
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


class AdultIncomeDataset(Dataset):
    """UCI Adult Income Dataset"""
    
    def __init__(self, data_dir='./data', split='train', train_ratio=0.7, val_ratio=0.15, random_seed=42):
        """
        Args:
            data_dir: Directory containing the dataset
            split: 'train', 'val', or 'test'
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            random_seed: Random seed for reproducibility
        """
        self.data_dir = data_dir
        self.split = split
        
        # Download and load data
        self._load_data(train_ratio, val_ratio, random_seed)
    
    def _download_data(self):
        """Download Adult dataset if not present"""
        import urllib.request
        
        os.makedirs(self.data_dir, exist_ok=True)
        
        urls = {
            'train': 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
            'test': 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'
        }
        
        for name, url in urls.items():
            filepath = os.path.join(self.data_dir, f'adult_{name}.csv')
            if not os.path.exists(filepath):
                print(f"Downloading {name} data...")
                urllib.request.urlretrieve(url, filepath)
    
    def _load_data(self, train_ratio, val_ratio, random_seed):
        """Load and preprocess Adult dataset"""
        self._download_data()
        
        # Column names
        columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                  'marital-status', 'occupation', 'relationship', 'race', 'sex',
                  'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
        
        # Load data
        train_path = os.path.join(self.data_dir, 'adult_train.csv')
        df = pd.read_csv(train_path, names=columns, skipinitialspace=True)
        
        # Clean data
        df = df.replace('?', np.nan).dropna()
        
        # Separate features and target
        X = df.drop('income', axis=1)
        y = df['income'].apply(lambda x: 1 if '>50K' in x else 0)
        
        # Encode categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        label_encoders = {}
        
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            label_encoders[col] = le
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(1-train_ratio), random_state=random_seed, stratify=y
        )
        
        val_ratio_adjusted = val_ratio / (val_ratio + (1 - train_ratio - val_ratio))
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(1-val_ratio_adjusted), random_state=random_seed, stratify=y_temp
        )
        
        # Standardize numerical features
        self.scaler = StandardScaler()
        
        if self.split == 'train':
            X_train = self.scaler.fit_transform(X_train)
            self.features = torch.FloatTensor(X_train)
            self.labels = torch.LongTensor(y_train.values)
        elif self.split == 'val':
            X_val = self.scaler.fit_transform(X_val)
            self.features = torch.FloatTensor(X_val)
            self.labels = torch.LongTensor(y_val.values)
        else:  # test
            X_test = self.scaler.fit_transform(X_test)
            self.features = torch.FloatTensor(X_test)
            self.labels = torch.LongTensor(y_test.values)
        
        self.input_dim = self.features.shape[1]
        self.num_classes = 2
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class CIFAR10Dataset:
    """CIFAR-10 Dataset Wrapper"""
    
    def __init__(self, data_dir='./data', split='train'):
        """
        Args:
            data_dir: Directory to store/load CIFAR-10
            split: 'train', 'val', or 'test'
        """
        self.data_dir = data_dir
        self.split = split
        
        # Data augmentation for training
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        
        # Load dataset
        if split in ['train', 'val']:
            full_dataset = datasets.CIFAR10(root=data_dir, train=True, 
                                           download=True, transform=self.transform)
            
            # Split train into train/val
            train_size = int(0.9 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
            
            self.dataset = train_dataset if split == 'train' else val_dataset
        else:  # test
            self.dataset = datasets.CIFAR10(root=data_dir, train=False, 
                                           download=True, transform=self.transform)
        
        self.input_dim = (3, 32, 32)
        self.num_classes = 10
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]


class PCamDataset(Dataset):
    """PatchCamelyon (PCam) Dataset"""
    
    def __init__(self, data_dir='./data', split='train'):
        """
        Args:
            data_dir: Directory containing PCam data
            split: 'train', 'val', or 'test'
        """
        # Note: PCam needs to be downloaded separately from:
        # https://github.com/basveeling/pcam
        
        self.data_dir = os.path.join(data_dir, 'pcam')
        self.split = split
        
        # Transform
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        # Load data (placeholder - actual implementation depends on downloaded format)
        self._load_data()
        
        self.input_dim = (3, 96, 96)
        self.num_classes = 2
    
    def _load_data(self):
        """Load PCam data - simplified version"""
        # This is a placeholder. In practice, you would load the actual HDF5 files
        # For now, create dummy data
        print(f"Note: PCam dataset needs to be downloaded separately.")
        print("Creating dummy data for demonstration...")
        
        if self.split == 'train':
            num_samples = 1000
        elif self.split == 'val':
            num_samples = 200
        else:
            num_samples = 200
        
        # Dummy data (replace with actual loading)
        self.images = np.random.randint(0, 255, (num_samples, 96, 96, 3), dtype=np.uint8)
        self.labels = np.random.randint(0, 2, num_samples, dtype=np.int64)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_dataloader(dataset_name, config, split='train'):
    """
    Get dataloader for specified dataset
    
    Args:
        dataset_name: 'adult', 'cifar10', or 'pcam'
        config: Configuration dictionary
        split: 'train', 'val', or 'test'
    
    Returns:
        DataLoader, input_dim, num_classes
    """
    data_dir = config['paths']['data_dir']
    batch_size = config['training']['batch_size']
    
    if dataset_name == 'adult':
        dataset = AdultIncomeDataset(
            data_dir=data_dir,
            split=split,
            train_ratio=config['data']['train_split'],
            val_ratio=config['data']['val_split'],
            random_seed=config['data']['random_seed']
        )
    elif dataset_name == 'cifar10':
        dataset = CIFAR10Dataset(data_dir=data_dir, split=split)
    elif dataset_name == 'pcam':
        dataset = PCamDataset(data_dir=data_dir, split=split)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Create dataloader
    shuffle = (split == 'train')
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=True
    )
    
    return dataloader, dataset.input_dim, dataset.num_classes
