# Analysis Notebook Template
# Use this to visualize and analyze your experiment results

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# Load results
with open('../results/all_results.json', 'r') as f:
    all_results = json.load(f)

df = pd.DataFrame(all_results)

# Display results table
print("Experiment Results:")
print(df[['dataset', 'architecture', 'test_accuracy', 'test_f1', 'num_parameters', 'training_time_minutes']])

# Visualization 1: Accuracy by Dataset and Architecture
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
pivot_acc = df.pivot(index='dataset', columns='architecture', values='test_accuracy')
pivot_acc.plot(kind='bar', ax=ax)
ax.set_title('Test Accuracy by Dataset and Architecture')
ax.set_ylabel('Accuracy')
ax.set_xlabel('Dataset')
ax.legend(title='Architecture')
plt.tight_layout()
plt.savefig('../results/accuracy_comparison.png', dpi=300)

# Visualization 2: F1-Score Comparison
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
pivot_f1 = df.pivot(index='dataset', columns='architecture', values='test_f1')
pivot_f1.plot(kind='bar', ax=ax)
ax.set_title('F1-Score by Dataset and Architecture')
ax.set_ylabel('F1-Score')
ax.set_xlabel('Dataset')
ax.legend(title='Architecture')
plt.tight_layout()
plt.savefig('../results/f1_comparison.png', dpi=300)

# Visualization 3: Parameter Count vs Performance
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
for dataset in df['dataset'].unique():
    subset = df[df['dataset'] == dataset]
    ax.scatter(subset['num_parameters'], subset['test_accuracy'], 
              label=dataset, s=100, alpha=0.7)

ax.set_xlabel('Number of Parameters')
ax.set_ylabel('Test Accuracy')
ax.set_title('Model Complexity vs Performance')
ax.legend()
ax.set_xscale('log')
plt.tight_layout()
plt.savefig('../results/params_vs_accuracy.png', dpi=300)

# Visualization 4: Training Time Analysis
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
pivot_time = df.pivot(index='dataset', columns='architecture', values='training_time_minutes')
pivot_time.plot(kind='bar', ax=ax)
ax.set_title('Training Time by Dataset and Architecture')
ax.set_ylabel('Training Time (minutes)')
ax.set_xlabel('Dataset')
ax.legend(title='Architecture')
plt.tight_layout()
plt.savefig('../results/training_time_comparison.png', dpi=300)

print("\nAnalysis visualizations saved to results/ directory")
