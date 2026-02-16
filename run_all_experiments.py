"""
Run all experiments (3 datasets × 3 architectures = 9 experiments)
"""
import os
import sys
import json
import pandas as pd
from train import main

# Experiment configurations
DATASETS = ['adult', 'cifar10', 'pcam']
ARCHITECTURES = ['mlp', 'cnn', 'attention']


def run_all_experiments(config_path='configs/config.yaml'):
    """Run all 9 experiments and collect results"""
    
    all_results = []
    
    print("\n" + "="*80)
    print("RUNNING ALL EXPERIMENTS")
    print(f"Total experiments: {len(DATASETS)} datasets × {len(ARCHITECTURES)} architectures = {len(DATASETS) * len(ARCHITECTURES)}")
    print("="*80 + "\n")
    
    for i, dataset in enumerate(DATASETS):
        for j, architecture in enumerate(ARCHITECTURES):
            exp_num = i * len(ARCHITECTURES) + j + 1
            total_exp = len(DATASETS) * len(ARCHITECTURES)
            
            print(f"\n{'='*80}")
            print(f"EXPERIMENT {exp_num}/{total_exp}: {dataset.upper()} + {architecture.upper()}")
            print(f"{'='*80}\n")
            
            try:
                # Run experiment
                results = main(
                    config_path=config_path,
                    override_dataset=dataset,
                    override_architecture=architecture
                )
                all_results.append(results)
                
            except Exception as e:
                print(f"\n❌ ERROR in experiment {dataset} + {architecture}:")
                print(f"   {str(e)}")
                # Create error result
                error_result = {
                    'dataset': dataset,
                    'architecture': architecture,
                    'error': str(e),
                    'test_accuracy': 0.0,
                    'test_f1': 0.0
                }
                all_results.append(error_result)
                continue
    
    # Create results summary
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED!")
    print("="*80)
    
    # Save all results
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Save to JSON
    all_results_path = os.path.join(results_dir, 'all_results.json')
    with open(all_results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nAll results saved to: {all_results_path}")
    
    # Create results table
    create_results_table(all_results, results_dir)
    
    return all_results


def create_results_table(all_results, results_dir):
    """Create and save results comparison table"""
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    # Select key columns
    columns = ['dataset', 'architecture', 'test_accuracy', 'test_f1', 
               'num_parameters', 'training_time_minutes']
    
    if all(col in df.columns for col in columns):
        table_df = df[columns].copy()
        
        # Format numbers
        table_df['test_accuracy'] = table_df['test_accuracy'].apply(lambda x: f"{x:.4f}")
        table_df['test_f1'] = table_df['test_f1'].apply(lambda x: f"{x:.4f}")
        table_df['num_parameters'] = table_df['num_parameters'].apply(lambda x: f"{x:,}")
        table_df['training_time_minutes'] = table_df['training_time_minutes'].apply(lambda x: f"{x:.2f}")
        
        # Rename columns
        table_df.columns = ['Dataset', 'Architecture', 'Accuracy', 'F1', 'Parameters', 'Time (min)']
        
        # Save to CSV
        csv_path = os.path.join(results_dir, 'results_table.csv')
        table_df.to_csv(csv_path, index=False)
        print(f"Results table saved to: {csv_path}")
        
        # Print table
        print("\n" + "="*80)
        print("RESULTS TABLE")
        print("="*80)
        print(table_df.to_string(index=False))
        print("="*80)
        
        # Create markdown table
        markdown_path = os.path.join(results_dir, 'results_table.md')
        with open(markdown_path, 'w') as f:
            f.write("# Experiment Results\n\n")
            f.write(table_df.to_markdown(index=False))
        print(f"\nMarkdown table saved to: {markdown_path}")
    
    return table_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run all benchmark experiments')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    run_all_experiments(config_path=args.config)
