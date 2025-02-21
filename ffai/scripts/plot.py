import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def load_metrics_files(checkpoint_dir):
    """Load all metrics files from checkpoint directory"""
    metrics_files = sorted(
        Path(checkpoint_dir).glob('checkpoint_*_metrics.json'),
        key=lambda x: int(x.stem.split('_')[1])
    )

    all_metrics = []
    for file in metrics_files:
        with open(file, 'r') as f:
            metrics = json.load(f)
            episode_num = int(file.stem.split('_')[1])
            metrics['episode'] = episode_num
            all_metrics.append(metrics)

    return all_metrics

def create_plots(metrics, output_dir):
    """Generate plots from metrics data"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Delete any existing plot files
    for plot_file in output_dir.glob('*.png'):
        plot_file.unlink(missing_ok=True)

    # Extract episode data into DataFrame
    episodes_data = []
    for m in metrics:
        for episode in m['recent_episodes']:
            episodes_data.append({
                'episode': episode['episode'],
                'wins': episode['season']['wins'],
                'avg_points': episode['season']['avg_points_per_week'],
                'standing': episode['season']['final_standing'] + 1,
                'draft_reward': episode['draft']['reward'],
                'season_reward': episode['season']['reward'],
                'loss': episode['model']['loss'],
                'running_reward': episode['model']['running_reward'],
                'total_spent': episode['draft']['total_spent'],
                'projected_points': episode['draft']['total_projected_points'],
            })

    df = pd.DataFrame(episodes_data)

    # Set style to a built-in matplotlib style
    plt.style.use('ggplot')  # Alternative built-in style

    # 1. Season Performance Plot
    plt.figure(figsize=(12, 6))
    plt.plot(df['episode'], df['wins'].rolling(100).mean(), label='Wins (100-ep avg)')
    plt.plot(df['episode'], df['standing'].rolling(100).mean(), label='Standing (100-ep avg)')
    plt.xlabel('Episode')
    plt.ylabel('Count')
    plt.title('Season Performance Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / 'season_performance.png')
    plt.close()

    # 2. Model Training Plot
    plt.figure(figsize=(12, 6))
    plt.plot(df['episode'], df['loss'].rolling(100).mean(), label='Loss (100-ep avg)')
    plt.plot(df['episode'], df['running_reward'].rolling(100).mean(), label='Running Reward (100-ep avg)')
    plt.xlabel('Episode')
    plt.ylabel('Value')
    plt.title('Model Training Metrics')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / 'training_metrics.png')
    plt.close()

    # 3. Draft Strategy Plot
    plt.figure(figsize=(12, 6))
    # plt.plot(df['episode'], df['total_spent'].rolling(100).mean(), label='Total Spent (100-ep avg)')
    plt.plot(df['episode'], df['projected_points'].rolling(100).mean(), label='Projected Points (100-ep avg)')
    plt.xlabel('Episode')
    plt.ylabel('Value')
    plt.title('Draft Strategy Evolution')
    plt.legend()
    plt.grid(True)  # Add grid lines for both major and minor ticks
    plt.savefig(output_dir / 'draft_strategy.png')
    plt.close()

    # 4. Points Distribution Box Plot
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, y='avg_points')
    plt.title('Distribution of Average Points per Week')
    plt.ylabel('Points')
    plt.savefig(output_dir / 'points_distribution.png')
    plt.close()

    # 5. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    correlation_matrix = df[[
        'wins', 'avg_points', 'standing', 'draft_reward',
        'season_reward', 'total_spent', 'projected_points'
    ]].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Between Metrics')
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_heatmap.png')
    plt.close()

def main():
    checkpoint_dir = Path(__file__).parents[1] / "checkpoints"
    output_dir = Path(__file__).parents[1] / "plots"

    # Load metrics
    metrics = load_metrics_files(checkpoint_dir)

    # Generate plots
    create_plots(metrics, output_dir)

if __name__ == "__main__":
    main()
