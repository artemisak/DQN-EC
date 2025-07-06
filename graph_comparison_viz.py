import matplotlib.pyplot as plt
import pandas as pd
import json
import seaborn as sns
import os

def load_results(json_file):
    """Load results from JSON file"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data['results'])

def plot_comparison_results(df, save_dir="comparison_plots"):
    """Create comprehensive plots comparing different algorithms"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # Get unique algorithms and epochs
    algorithms = df['Algorithm'].unique()
    epochs = sorted(df['Epoch'].unique())
    
    # Create figure with subplots for each metric
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Graph Algorithm Comparison - Loss Components Over Epochs', fontsize=16)
    
    metrics = [
        ('First Term (weight1 * labels_kl)', 'First Term Std', 'First Term Loss'),
        ('Second Term (weight2 * values_mse)', 'Second Term Std', 'Second Term Loss'),
        ('Graph Loss (1e-6 * l1_loss)', 'Graph Loss Std', 'Graph Sparsity Loss'),
        ('Total Loss', 'Total Loss Std', 'Total Loss')
    ]
    
    for idx, (metric, std_metric, title) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        for algo in algorithms:
            algo_data = df[df['Algorithm'] == algo]
            values = algo_data[metric].values
            stds = algo_data[std_metric].values
            
            # Plot mean with error bars
            ax.errorbar(epochs, values, yerr=stds, label=algo, marker='o', 
                       capsize=5, capthick=2, linewidth=2, markersize=8)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss Value', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set y-axis to log scale if values vary greatly
        if metric == 'Graph Loss (1e-6 * l1_loss)':
            ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_components_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create individual plots for better detail
    for metric, std_metric, title in metrics:
        plt.figure(figsize=(10, 6))
        
        for algo in algorithms:
            algo_data = df[df['Algorithm'] == algo]
            values = algo_data[metric].values
            stds = algo_data[std_metric].values
            
            plt.errorbar(epochs, values, yerr=stds, label=algo, marker='o', 
                        capsize=5, capthick=2, linewidth=2, markersize=8)
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss Value', fontsize=12)
        plt.title(f'{title} - Detailed Comparison', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if metric == 'Graph Loss (1e-6 * l1_loss)':
            plt.yscale('log')
        
        filename = title.lower().replace(' ', '_') + '_detailed.png'
        plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
        plt.show()
    
    # Create heatmap showing final epoch performance
    final_epoch = max(epochs)
    final_data = df[df['Epoch'] == final_epoch]
    
    # Prepare data for heatmap
    heatmap_data = []
    for algo in algorithms:
        algo_final = final_data[final_data['Algorithm'] == algo]
        heatmap_data.append([
            algo_final['First Term (weight1 * labels_kl)'].values[0],
            algo_final['Second Term (weight2 * values_mse)'].values[0],
            algo_final['Graph Loss (1e-6 * l1_loss)'].values[0],
            algo_final['Total Loss'].values[0]
        ])
    
    # Create heatmap
    plt.figure(figsize=(10, 6))
    heatmap_df = pd.DataFrame(heatmap_data, 
                             index=algorithms,
                             columns=['First Term', 'Second Term', 'Graph Loss', 'Total Loss'])
    
    # Normalize each column for better visualization
    heatmap_normalized = heatmap_df.div(heatmap_df.max(axis=0), axis=1)
    
    sns.heatmap(heatmap_normalized, annot=heatmap_df.values, fmt='.6f', 
                cmap='RdYlGn_r', cbar_kws={'label': 'Normalized Loss'})
    plt.title(f'Final Performance Comparison (Epoch {final_epoch})', fontsize=14)
    plt.xlabel('Loss Component', fontsize=12)
    plt.ylabel('Algorithm', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'final_performance_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create convergence speed plot
    plt.figure(figsize=(10, 6))
    
    for algo in algorithms:
        algo_data = df[df['Algorithm'] == algo]
        total_losses = algo_data['Total Loss'].values
        
        # Calculate relative improvement from epoch 1
        relative_improvement = (total_losses[0] - total_losses) / total_losses[0] * 100
        plt.plot(epochs, relative_improvement, label=algo, marker='o', linewidth=2, markersize=8)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Relative Improvement (%)', fontsize=12)
    plt.title('Convergence Speed Comparison', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'convergence_speed.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    for algo in algorithms:
        algo_data = df[df['Algorithm'] == algo]
        final_algo = final_data[final_data['Algorithm'] == algo]
        
        print(f"\n{algo}:")
        print(f"  Final Total Loss: {final_algo['Total Loss'].values[0]:.6f} ± {final_algo['Total Loss Std'].values[0]:.6f}")
        print(f"  Best Epoch: {algo_data.loc[algo_data['Total Loss'].idxmin(), 'Epoch']}")
        print(f"  Best Total Loss: {algo_data['Total Loss'].min():.6f}")
        
        # Calculate convergence rate
        total_losses = algo_data['Total Loss'].values
        improvement = (total_losses[0] - total_losses[-1]) / total_losses[0] * 100
        print(f"  Total Improvement: {improvement:.2f}%")

def create_latex_table(df, output_file="comparison_table.tex"):
    """Create LaTeX table from results"""
    
    # Select columns for LaTeX table
    display_df = df[['Algorithm', 'Epoch', 'First Term (weight1 * labels_kl)', 
                     'Second Term (weight2 * values_mse)', 'Graph Loss (1e-6 * l1_loss)', 'Total Loss']]
    
    # Round values for better display
    for col in display_df.columns[2:]:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.6f}")
    
    # Create LaTeX table
    latex_table = display_df.to_latex(index=False, 
                                     column_format='l' + 'c' * (len(display_df.columns) - 1),
                                     caption="Graph Algorithm Comparison Results",
                                     label="tab:graph_comparison")
    
    # Save to file
    with open(output_file, 'w') as f:
        f.write(latex_table)
    
    print(f"\nLaTeX table saved to {output_file}")

if __name__ == "__main__":
    # Load the most recent results file
    import glob
    
    json_files = glob.glob('graph_comparison_results_*.json')
    if not json_files:
        print("No results files found. Please run the comparison script first.")
    else:
        latest_file = max(json_files)
        print(f"Loading results from: {latest_file}")
        
        df = load_results(latest_file)
        plot_comparison_results(df)
        create_latex_table(df)
