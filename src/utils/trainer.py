"""
Training utilities: trainer, evaluator, metrics
"""
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class Trainer:
    """Training manager for neural networks"""
    
    def __init__(self, model, train_loader, val_loader, config, device):
        """
        Args:
            model: Neural network model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Configuration dictionary
            device: Device to train on
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        optimizer_name = config['training']['optimizer'].lower()
        lr = config['training']['learning_rate']
        weight_decay = config['training']['weight_decay']
        
        if optimizer_name == 'adam':
            self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            self.optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Early stopping
        self.early_stopping = config['training']['early_stopping']['enabled']
        self.patience = config['training']['early_stopping']['patience']
        self.min_delta = config['training']['early_stopping']['min_delta']
        
        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0
        
        # Paths
        self.checkpoint_dir = config['paths']['checkpoint_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        # Compute metrics
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Compute metrics
        val_loss = running_loss / len(self.val_loader)
        val_acc = accuracy_score(all_labels, all_preds)
        
        return val_loss, val_acc
    
    def train(self, num_epochs):
        """Train for multiple epochs"""
        print(f"\nTraining for {num_epochs} epochs...")
        print(f"Model parameters: {self.model.get_param_count():,}")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # Validate
            val_loss, val_acc = self.validate()
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.save_checkpoint('best_model.pth')
                self.epochs_without_improvement = 0
                print(f"✓ New best model saved! Val Acc: {val_acc:.4f}")
            else:
                self.epochs_without_improvement += 1
            
            # Early stopping
            if self.early_stopping and self.epochs_without_improvement >= self.patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time/60:.2f} minutes")
        print(f"Best validation accuracy: {self.best_val_acc:.4f}")
        
        return training_time
    
    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'best_val_acc': self.best_val_acc,
        }, filepath)
    
    def load_checkpoint(self, filename):
        """Load model checkpoint"""
        filepath = os.path.join(self.checkpoint_dir, filename)
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint
    
    def plot_training_curves(self, save_path=None):
        """Plot training and validation curves"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss curves
        axes[0].plot(self.train_losses, label='Train Loss')
        axes[0].plot(self.val_losses, label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy curves
        axes[1].plot(self.train_accs, label='Train Acc')
        axes[1].plot(self.val_accs, label='Val Acc')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


class Evaluator:
    """Model evaluator with comprehensive metrics"""
    
    def __init__(self, model, test_loader, device):
        self.model = model
        self.test_loader = test_loader
        self.device = device
    
    def evaluate(self):
        """Evaluate model on test set"""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(self.test_loader, desc='Evaluating'):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Compute metrics
        accuracy = accuracy_score(all_labels, all_preds)
        
        # Handle binary vs multiclass F1
        if len(np.unique(all_labels)) == 2:
            f1 = f1_score(all_labels, all_preds, average='binary')
            precision = precision_score(all_labels, all_preds, average='binary')
            recall = recall_score(all_labels, all_preds, average='binary')
        else:
            f1 = f1_score(all_labels, all_preds, average='weighted')
            precision = precision_score(all_labels, all_preds, average='weighted')
            recall = recall_score(all_labels, all_preds, average='weighted')
        
        cm = confusion_matrix(all_labels, all_preds)
        
        results = {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': cm,
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs
        }
        
        return results
    
    def print_results(self, results):
        """Print evaluation results"""
        print("\n" + "="*50)
        print("TEST SET RESULTS")
        print("="*50)
        print(f"Accuracy:  {results['accuracy']:.4f}")
        print(f"F1 Score:  {results['f1']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall:    {results['recall']:.4f}")
        print("="*50 + "\n")
    
    def plot_confusion_matrix(self, cm, class_names=None, save_path=None):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        # Generate default class names if not provided
        if class_names is None:
            class_names = [f'Class {i}' for i in range(len(cm))]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
