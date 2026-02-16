"""
Main training script for Deep Learning Benchmark
Run experiments with different datasets and architectures
"""
import os
import sys
import yaml
import torch
import argparse
import json
from datetime import datetime

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.datasets.data_loader import get_dataloader
from src.models.mlp import MLP
from src.models.cnn import CNN
from src.models.attention import VisionTransformer, AttentionMLP
from src.utils.trainer import Trainer, Evaluator


def load_config(config_path='configs/config.yaml'):
    """Load configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_model(architecture, input_dim, num_classes, config):
    """
    Create model based on architecture name
    
    Args:
        architecture: 'mlp', 'cnn', or 'attention'
        input_dim: Input dimension
        num_classes: Number of output classes
        config: Configuration dictionary
    
    Returns:
        model: PyTorch model
    """
    arch_config = config['model'][architecture]
    
    if architecture == 'mlp':
        model = MLP(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dims=arch_config['hidden_dims'],
            dropout=arch_config['dropout'],
            use_batch_norm=arch_config['use_batch_norm']
        )
    
    elif architecture == 'cnn':
        model = CNN(
            input_dim=input_dim,
            num_classes=num_classes,
            conv_channels=arch_config['conv_channels'],
            kernel_sizes=arch_config['kernel_sizes'],
            pool_sizes=arch_config['pool_sizes'],
            fc_dims=arch_config['fc_dims'],
            dropout=arch_config['dropout'],
            use_batch_norm=arch_config['use_batch_norm']
        )
    
    elif architecture == 'attention':
        # Choose attention model based on input type
        if isinstance(input_dim, tuple) and len(input_dim) == 3:
            # Image data - use Vision Transformer
            model = VisionTransformer(
                input_dim=input_dim,
                num_classes=num_classes,
                d_model=arch_config['d_model'],
                num_heads=arch_config['num_heads'],
                num_layers=arch_config['num_layers'],
                dropout=arch_config['dropout'],
                fc_dims=arch_config['fc_dims']
            )
        else:
            # Tabular data - use Attention MLP
            model = AttentionMLP(
                input_dim=input_dim,
                num_classes=num_classes,
                d_model=arch_config['d_model'],
                num_heads=arch_config['num_heads'],
                dropout=arch_config['dropout'],
                fc_dims=arch_config['fc_dims']
            )
    
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    return model


def main(config_path='configs/config.yaml', override_dataset=None, override_architecture=None):
    """
    Main training function
    
    Args:
        config_path: Path to configuration file
        override_dataset: Override dataset from config (optional)
        override_architecture: Override architecture from config (optional)
    """
    # Load configuration
    config = load_config(config_path)
    
    # Override if specified
    if override_dataset:
        config['dataset'] = override_dataset
    if override_architecture:
        config['architecture'] = override_architecture
    
    dataset_name = config['dataset']
    architecture = config['architecture']
    
    print("="*60)
    print(f"Deep Learning Benchmark Experiment")
    print("="*60)
    print(f"Dataset: {dataset_name}")
    print(f"Architecture: {architecture}")
    print(f"Device: {config['device']}")
    print("="*60)
    
    # Setup device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    for path_key in ['data_dir', 'results_dir', 'checkpoint_dir', 'log_dir']:
        os.makedirs(config['paths'][path_key], exist_ok=True)
    
    # Load data
    print("\nLoading datasets...")
    train_loader, input_dim, num_classes = get_dataloader(dataset_name, config, split='train')
    val_loader, _, _ = get_dataloader(dataset_name, config, split='val')
    test_loader, _, _ = get_dataloader(dataset_name, config, split='test')
    
    print(f"Input dimension: {input_dim}")
    print(f"Number of classes: {num_classes}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print("\nCreating model...")
    model = get_model(architecture, input_dim, num_classes, config)
    print(f"Model created with {model.get_param_count():,} parameters")
    
    # Training
    print("\nInitializing trainer...")
    trainer = Trainer(model, train_loader, val_loader, config, device)
    
    num_epochs = config['training']['num_epochs']
    training_time = trainer.train(num_epochs)
    
    # Save training curves
    results_dir = config['paths']['results_dir']
    exp_name = f"{dataset_name}_{architecture}"
    curves_path = os.path.join(results_dir, f"{exp_name}_curves.png")
    trainer.plot_training_curves(save_path=curves_path)
    print(f"\nTraining curves saved to: {curves_path}")
    
    # Evaluation
    print("\nEvaluating on test set...")
    trainer.load_checkpoint('best_model.pth')
    evaluator = Evaluator(model, test_loader, device)
    results = evaluator.evaluate()
    evaluator.print_results(results)
    
    # Save confusion matrix
    cm_path = os.path.join(results_dir, f"{exp_name}_confusion_matrix.png")
    evaluator.plot_confusion_matrix(results['confusion_matrix'], save_path=cm_path)
    print(f"Confusion matrix saved to: {cm_path}")
    
    # Save results to JSON
    results_summary = {
        'dataset': dataset_name,
        'architecture': architecture,
        'num_parameters': model.get_param_count(),
        'training_time_seconds': training_time,
        'training_time_minutes': training_time / 60,
        'num_epochs_trained': len(trainer.train_losses),
        'best_val_accuracy': trainer.best_val_acc,
        'test_accuracy': results['accuracy'],
        'test_f1': results['f1'],
        'test_precision': results['precision'],
        'test_recall': results['recall'],
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    results_json_path = os.path.join(results_dir, f"{exp_name}_results.json")
    with open(results_json_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\nResults saved to: {results_json_path}")
    
    print("\n" + "="*60)
    print("Experiment completed successfully!")
    print("="*60)
    
    return results_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deep Learning Benchmark')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--dataset', type=str, choices=['adult', 'cifar10', 'pcam'],
                       help='Dataset to use (overrides config)')
    parser.add_argument('--architecture', type=str, choices=['mlp', 'cnn', 'attention'],
                       help='Architecture to use (overrides config)')
    
    args = parser.parse_args()
    
    main(
        config_path=args.config,
        override_dataset=args.dataset,
        override_architecture=args.architecture
    )
