import matplotlib.pyplot as plt
import pandas as pd
import json
import seaborn as sns
import numpy as np
import os
from scipy import stats


def load_and_process_results(json_file):
    """Load results from JSON file and process into aggregated DataFrame"""
    with open(json_file, 'r') as f:
        data = json.load(f)

    raw_data = data['raw_data']
    parameters = data['parameters']

    # Process the data into a structured format
    processed_data = []

    for algorithm, epochs_data in raw_data.items():
        for epoch_str, epoch_data in epochs_data.items():
            epoch = int(epoch_str)
            runs_data = epoch_data['runs']

            # Extract values from all runs
            first_terms = [run['first_term'] for run in runs_data]
            second_terms = [run['second_term'] for run in runs_data]
            graph_losses = [run['graph_loss'] for run in runs_data]
            total_losses = [run['total_loss'] for run in runs_data]

            # Filter outliers using IQR method for each metric
            def filter_outliers(values):
                """Remove outliers using IQR method"""
                if len(values) < 3:  # Not enough data to filter outliers
                    return values

                q1 = np.percentile(values, 25)
                q3 = np.percentile(values, 75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr

                filtered = [v for v in values if lower_bound <= v <= upper_bound]

                # If we filter out too many values, use original
                if len(filtered) < len(values) * 0.5:
                    return values

                return filtered

            # Apply outlier filtering
            first_terms_filtered = filter_outliers(first_terms)
            second_terms_filtered = filter_outliers(second_terms)
            graph_losses_filtered = filter_outliers(graph_losses)
            total_losses_filtered = filter_outliers(total_losses)

            # Calculate statistics
            processed_data.append({
                'Algorithm': algorithm,
                'Epoch': epoch,
                'First Term Mean': np.mean(first_terms_filtered),
                'First Term Std': np.std(first_terms_filtered),
                'Second Term Mean': np.mean(second_terms_filtered),
                'Second Term Std': np.std(second_terms_filtered),
                'Graph Loss Mean': np.mean(graph_losses_filtered),
                'Graph Loss Std': np.std(graph_losses_filtered),
                'Total Loss Mean': np.mean(total_losses_filtered),
                'Total Loss Std': np.std(total_losses_filtered),
                'Num Runs': len(runs_data),
                'Num Runs After Filtering': len(total_losses_filtered)
            })

    return pd.DataFrame(processed_data), parameters


def plot_comparison_results(df, parameters, save_dir="comparison_plots"):
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
    fig.suptitle(f'Graph Algorithm Comparison - Loss Components Over Epochs\n'
                 f'(Learning Rate: {parameters["learning_rate"]}, Runs: {parameters["num_runs"]})',
                 fontsize=16)

    metrics = [
        ('First Term Mean', 'First Term Std', 'First Term Loss (KL Divergence)'),
        ('Second Term Mean', 'Second Term Std', 'Second Term Loss (MSE)'),
        ('Graph Loss Mean', 'Graph Loss Std', 'Graph Sparsity Loss'),
        ('Total Loss Mean', 'Total Loss Std', 'Total Loss')
    ]

    for idx, (metric, std_metric, title) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]

        for algo in algorithms:
            algo_data = df[df['Algorithm'] == algo].sort_values('Epoch')
            values = algo_data[metric].values
            stds = algo_data[std_metric].values

            # Plot mean with error bars
            ax.errorbar(epochs, values, yerr=stds, label=algo, marker='o',
                        capsize=5, capthick=2, linewidth=2, markersize=8)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss Value', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Use log scale for better visualization if values span multiple orders of magnitude
        if metric in ['Graph Loss Mean']:
            ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_components_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # Create individual plots for better detail
    for metric, std_metric, title in metrics:
        plt.figure(figsize=(10, 6))

        for algo in algorithms:
            algo_data = df[df['Algorithm'] == algo].sort_values('Epoch')
            values = algo_data[metric].values
            stds = algo_data[std_metric].values

            plt.errorbar(epochs, values, yerr=stds, label=algo, marker='o',
                         capsize=5, capthick=2, linewidth=2, markersize=8)

        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss Value', fontsize=12)
        plt.title(f'{title} - Detailed Comparison', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)

        if metric == 'Graph Loss Mean':
            plt.yscale('log')

        filename = title.lower().replace(' ', '_').replace('(', '').replace(')', '') + '_detailed.png'
        plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
        plt.show()

    # Create heatmap showing final epoch performance
    final_epoch = max(epochs)
    final_data = df[df['Epoch'] == final_epoch]

    # Prepare data for heatmap
    heatmap_data = []
    for algo in algorithms:
        algo_final = final_data[final_data['Algorithm'] == algo]
        if not algo_final.empty:
            heatmap_data.append([
                algo_final['First Term Mean'].values[0],
                algo_final['Second Term Mean'].values[0],
                algo_final['Graph Loss Mean'].values[0],
                algo_final['Total Loss Mean'].values[0]
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
        algo_data = df[df['Algorithm'] == algo].sort_values('Epoch')
        total_losses = algo_data['Total Loss Mean'].values

        # Calculate relative improvement from epoch 1
        if len(total_losses) > 0:
            relative_improvement = (total_losses[0] - total_losses) / total_losses[0] * 100
            plt.plot(epochs[:len(relative_improvement)], relative_improvement,
                     label=algo, marker='o', linewidth=2, markersize=8)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Relative Improvement (%)', fontsize=12)
    plt.title('Convergence Speed Comparison', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'convergence_speed.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # Create stability plot (standard deviation over epochs)
    plt.figure(figsize=(10, 6))

    for algo in algorithms:
        algo_data = df[df['Algorithm'] == algo].sort_values('Epoch')
        total_stds = algo_data['Total Loss Std'].values

        plt.plot(epochs[:len(total_stds)], total_stds,
                 label=algo, marker='o', linewidth=2, markersize=8)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Standard Deviation', fontsize=12)
    plt.title('Training Stability Comparison (Lower is Better)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.savefig(os.path.join(save_dir, 'training_stability.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    for algo in algorithms:
        algo_data = df[df['Algorithm'] == algo]
        final_algo = final_data[final_data['Algorithm'] == algo]

        if not final_algo.empty:
            print(f"\n{algo}:")
            print(
                f"  Final Total Loss: {final_algo['Total Loss Mean'].values[0]:.6f} ± {final_algo['Total Loss Std'].values[0]:.6f}")
            print(f"  Best Epoch: {algo_data.loc[algo_data['Total Loss Mean'].idxmin(), 'Epoch']}")
            print(f"  Best Total Loss: {algo_data['Total Loss Mean'].min():.6f}")

            # Calculate convergence rate
            total_losses = algo_data['Total Loss Mean'].values
            if len(total_losses) > 0:
                improvement = (total_losses[0] - total_losses[-1]) / total_losses[0] * 100
                print(f"  Total Improvement: {improvement:.2f}%")

            # Report outlier filtering impact
            total_runs = algo_data['Num Runs'].sum()
            filtered_runs = algo_data['Num Runs After Filtering'].sum()
            print(f"  Outliers Filtered: {total_runs - filtered_runs}/{total_runs} runs")


def create_latex_table(df, output_file="comparison_table.tex"):
    """Create LaTeX table from results"""

    # Select final epoch for the table
    final_epoch = df['Epoch'].max()
    final_df = df[df['Epoch'] == final_epoch].copy()

    # Format the display
    display_df = final_df[['Algorithm', 'First Term Mean', 'Second Term Mean',
                           'Graph Loss Mean', 'Total Loss Mean']].copy()

    # Round values and add standard deviations
    for col in ['First Term Mean', 'Second Term Mean', 'Graph Loss Mean', 'Total Loss Mean']:
        std_col = col.replace('Mean', 'Std')
        display_df[col] = final_df.apply(lambda x: f"{x[col]:.6f} ± {x[std_col]:.6f}", axis=1)

    # Rename columns for LaTeX
    display_df.columns = ['Algorithm', 'First Term', 'Second Term', 'Graph Loss', 'Total Loss']

    # Create LaTeX table
    latex_table = display_df.to_latex(index=False,
                                      column_format='l' + 'c' * (len(display_df.columns) - 1),
                                      caption=f"Graph Algorithm Comparison Results at Epoch {final_epoch}",
                                      label="tab:graph_comparison",
                                      escape=False)

    # Save to file
    with open(output_file, 'w') as f:
        f.write(latex_table)

    print(f"\nLaTeX table saved to {output_file}")


def create_performance_ranking(df, save_dir="comparison_plots"):
    """Create a bar chart showing algorithm ranking by final performance"""

    final_epoch = df['Epoch'].max()
    final_data = df[df['Epoch'] == final_epoch].copy()

    # Sort by total loss
    final_data = final_data.sort_values('Total Loss Mean')

    plt.figure(figsize=(10, 6))

    # Create bar chart
    bars = plt.bar(range(len(final_data)), final_data['Total Loss Mean'],
                   yerr=final_data['Total Loss Std'], capsize=5)

    # Color bars by performance (best = green, worst = red)
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(bars)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    plt.xticks(range(len(final_data)), final_data['Algorithm'], rotation=45, ha='right')
    plt.xlabel('Algorithm', fontsize=12)
    plt.ylabel('Final Total Loss', fontsize=12)
    plt.title(f'Algorithm Performance Ranking at Epoch {final_epoch}', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'performance_ranking.png'), dpi=300, bbox_inches='tight')
    plt.show()


def perform_statistical_comparison(raw_data, df, epoch=30, save_dir="comparison_plots"):
    """Perform two-sample t-tests between DAL and its closest competitor at specified epoch"""

    print("\n" + "=" * 80)
    print(f"STATISTICAL COMPARISON AT EPOCH {epoch}")
    print("=" * 80)

    # Check if DAL exists and epoch 30 data is available
    if 'DAL' not in raw_data or str(epoch) not in raw_data['DAL']:
        print(f"DAL data not found for epoch {epoch}")
        return

    # Get DAL's performance at epoch 30
    dal_epoch_data = df[(df['Algorithm'] == 'DAL') & (df['Epoch'] == epoch)]
    if dal_epoch_data.empty:
        print(f"No DAL data found for epoch {epoch}")
        return

    dal_total_loss = dal_epoch_data['Total Loss Mean'].values[0]

    # Find closest competitor based on total loss at epoch 30
    other_algos_epoch = df[(df['Algorithm'] != 'DAL') & (df['Epoch'] == epoch)].copy()
    if other_algos_epoch.empty:
        print("No other algorithms found for comparison")
        return

    other_algos_epoch['Diff_from_DAL'] = abs(other_algos_epoch['Total Loss Mean'] - dal_total_loss)
    closest_competitor = other_algos_epoch.loc[other_algos_epoch['Diff_from_DAL'].idxmin(), 'Algorithm']

    print(f"\nDAL Total Loss at Epoch {epoch}: {dal_total_loss:.6f}")
    print(f"Closest Competitor: {closest_competitor}")
    print(f"{closest_competitor} Total Loss at Epoch {epoch}: "
          f"{other_algos_epoch[other_algos_epoch['Algorithm'] == closest_competitor]['Total Loss Mean'].values[0]:.6f}")

    # Get raw data for both algorithms at epoch 30
    dal_runs = raw_data['DAL'][str(epoch)]['runs']
    competitor_runs = raw_data[closest_competitor][str(epoch)]['runs']

    # Extract values for each metric
    metrics = [
        ('first_term', 'First Term (KL Divergence)'),
        ('second_term', 'Second Term (MSE)'),
        ('graph_loss', 'Graph Loss (Sparsity)'),
        ('total_loss', 'Total Loss')
    ]

    print(f"\n{'=' * 60}")
    print(f"Two-Sample t-tests: DAL vs {closest_competitor}")
    print(f"{'=' * 60}")
    print(f"Number of runs: DAL={len(dal_runs)}, {closest_competitor}={len(competitor_runs)}")

    if len(dal_runs) < 3 or len(competitor_runs) < 3:
        print("\nWARNING: Sample size is very small (< 3). Results should be interpreted with caution.")
        print("Consider collecting more runs for more reliable statistical inference.")

    if len(dal_runs) < 2 or len(competitor_runs) < 2:
        print("\nERROR: Cannot perform t-test with fewer than 2 samples per group.")
        print("Please collect more runs before performing statistical comparison.")
        return

    results_summary = []

    for metric_key, metric_name in metrics:
        dal_values = [run[metric_key] for run in dal_runs]
        competitor_values = [run[metric_key] for run in competitor_runs]

        # Perform two-sample t-test
        # Use Welch's t-test (equal_var=False) as it's more robust for small samples
        t_stat, p_value = stats.ttest_ind(dal_values, competitor_values, equal_var=False)

        # Calculate effect size (Cohen's d)
        dal_mean = np.mean(dal_values)
        dal_std = np.std(dal_values, ddof=1)
        comp_mean = np.mean(competitor_values)
        comp_std = np.std(competitor_values, ddof=1)

        # Pooled standard deviation
        pooled_std = np.sqrt(((len(dal_values) - 1) * dal_std ** 2 +
                              (len(competitor_values) - 1) * comp_std ** 2) /
                             (len(dal_values) + len(competitor_values) - 2))

        cohen_d = (dal_mean - comp_mean) / pooled_std if pooled_std > 0 else 0

        print(f"\n{metric_name}:")
        print(f"  DAL:     mean={dal_mean:.6f}, std={dal_std:.6f}")
        print(f"  {closest_competitor}: mean={comp_mean:.6f}, std={comp_std:.6f}")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_value:.4f}")
        print(f"  Cohen's d: {cohen_d:.4f}")

        # Interpret results
        if p_value < 0.05:
            better = "DAL" if dal_mean < comp_mean else closest_competitor
            print(f"  Result: Statistically significant difference (p < 0.05)")
            print(f"          {better} performs significantly better")
        else:
            print(f"  Result: No statistically significant difference (p >= 0.05)")

        results_summary.append({
            'Metric': metric_name,
            'p-value': p_value,
            'Significant': p_value < 0.05,
            'Cohen\'s d': cohen_d
        })

    # Create visualization of the comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Statistical Comparison: DAL vs {closest_competitor} at Epoch {epoch}', fontsize=16)

    for idx, (metric_key, metric_name) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]

        dal_values = [run[metric_key] for run in dal_runs]
        competitor_values = [run[metric_key] for run in competitor_runs]

        # Create box plots
        bp = ax.boxplot([dal_values, competitor_values],
                        tick_labels=['DAL', closest_competitor],
                        patch_artist=True,
                        showmeans=True,
                        meanprops=dict(marker='D', markerfacecolor='red', markersize=8))

        # Color the boxes
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightgreen')

        # Add individual points
        x1 = np.random.normal(1, 0.04, len(dal_values))
        x2 = np.random.normal(2, 0.04, len(competitor_values))
        ax.scatter(x1, dal_values, alpha=0.5, s=30, color='blue')
        ax.scatter(x2, competitor_values, alpha=0.5, s=30, color='green')

        # Add p-value to plot
        p_val = results_summary[idx]['p-value']
        sig_text = "p < 0.05" if p_val < 0.05 else f"p = {p_val:.3f}"
        ax.text(0.5, 0.95, sig_text, transform=ax.transAxes,
                ha='center', va='top', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='wheat' if p_val < 0.05 else 'lightgray'))

        ax.set_ylabel('Loss Value', fontsize=10)
        ax.set_title(metric_name, fontsize=12)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'statistical_comparison_epoch{epoch}.png'),
                dpi=300, bbox_inches='tight')
    plt.show()

    # Summary table
    print(f"\n{'=' * 60}")
    print("SUMMARY OF STATISTICAL TESTS")
    print(f"{'=' * 60}")
    summary_df = pd.DataFrame(results_summary)
    print(summary_df.to_string(index=False))

    print("\nEffect Size Interpretation (Cohen's d):")
    print("  |d| < 0.2: Negligible")
    print("  0.2 ≤ |d| < 0.5: Small")
    print("  0.5 ≤ |d| < 0.8: Medium")
    print("  |d| ≥ 0.8: Large")


if __name__ == "__main__":
    # Load the most recent results file
    import glob

    json_files = glob.glob('graph_comparison_raw_*.json')
    if not json_files:
        print("No results files found. Please run the comparison script first.")
    else:
        latest_file = max(json_files)
        print(f"Loading results from: {latest_file}")

        # Load raw data for statistical tests
        with open(latest_file, 'r') as f:
            raw_json = json.load(f)

        df, parameters = load_and_process_results(latest_file)

        print(f"\nLoaded data for {len(df['Algorithm'].unique())} algorithms")
        print(f"Epochs tracked: {sorted(df['Epoch'].unique())}")
        print(f"Number of runs per configuration: {parameters['num_runs']}")

        save_dir = "comparison_plots"
        plot_comparison_results(df, parameters, save_dir)
        create_performance_ranking(df, save_dir)
        create_latex_table(df)

        # Perform statistical comparison at epoch 30
        perform_statistical_comparison(raw_json['raw_data'], df, epoch=30, save_dir=save_dir)