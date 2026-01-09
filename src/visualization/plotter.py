"""
Plotting utilities for visualization
"""
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class Plotter:
    """Plotting utilities for MARL system"""

    @staticmethod
    def plot_training_curves(training_data: Dict, algorithm_name: str = "RL Agent"):
        """Plot training curves using Plotly"""
        rewards = training_data.get('rewards', [])
        episode_lengths = training_data.get('episode_lengths', [])

        if not rewards:
            return None

        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Episode Rewards', 'Episode Lengths'),
            vertical_spacing=0.12
        )

        # Smooth rewards
        window = min(100, len(rewards) // 10) if len(rewards) > 10 else 1
        if window > 1:
            smoothed_rewards = np.convolve(rewards, np.ones(window) / window, mode='valid')
            x_smooth = list(range(len(smoothed_rewards)))
        else:
            smoothed_rewards = rewards
            x_smooth = list(range(len(rewards)))

        # Rewards plot
        fig.add_trace(
            go.Scatter(x=list(range(len(rewards))), y=rewards,
                      mode='lines', name='Raw Rewards',
                      line=dict(color='lightblue', width=1),
                      opacity=0.5),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=x_smooth, y=smoothed_rewards,
                      mode='lines', name='Smoothed Rewards',
                      line=dict(color='blue', width=2)),
            row=1, col=1
        )

        # Episode lengths
        if episode_lengths:
            fig.add_trace(
                go.Scatter(x=list(range(len(episode_lengths))), y=episode_lengths,
                          mode='lines', name='Episode Length',
                          line=dict(color='green', width=2)),
                row=2, col=1
            )

        # Update layout
        fig.update_xaxes(title_text="Episode", row=2, col=1)
        fig.update_yaxes(title_text="Reward", row=1, col=1)
        fig.update_yaxes(title_text="Length", row=2, col=1)

        fig.update_layout(
            title_text=f"{algorithm_name} - Training Progress",
            height=600,
            showlegend=True
        )

        return fig

    @staticmethod
    def plot_comparison(comparison_df: pd.DataFrame):
        """Plot algorithm comparison"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Mean Reward', 'Completion Rate', 'Mean Distance', 'Efficiency Score'),
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )

        algorithms = comparison_df['Algorithm'].tolist()

        # Mean Reward
        fig.add_trace(
            go.Bar(x=algorithms, y=comparison_df['mean_reward'],
                  name='Mean Reward', marker_color='blue'),
            row=1, col=1
        )

        # Completion Rate
        fig.add_trace(
            go.Bar(x=algorithms, y=comparison_df['mean_completion_rate'],
                  name='Completion Rate', marker_color='green'),
            row=1, col=2
        )

        # Mean Distance
        fig.add_trace(
            go.Bar(x=algorithms, y=comparison_df['mean_distance'],
                  name='Mean Distance', marker_color='orange'),
            row=2, col=1
        )

        # Efficiency Score
        fig.add_trace(
            go.Bar(x=algorithms, y=comparison_df['efficiency_score'],
                  name='Efficiency Score', marker_color='purple'),
            row=2, col=2
        )

        fig.update_layout(
            title_text="Algorithm Performance Comparison",
            height=700,
            showlegend=False
        )

        return fig

    @staticmethod
    def plot_warehouse_layout(env_state: Dict):
        """Plot warehouse layout with LGVs, pallets, and shelves"""
        fig = go.Figure()

        # Get environment state
        lgvs = env_state['lgvs']
        pallets = env_state['pallets']
        shelves = env_state['shelves']
        grid = env_state['grid']

        # Plot shelves
        shelf_x = [s['position'][0] for s in shelves]
        shelf_y = [s['position'][1] for s in shelves]

        fig.add_trace(go.Scatter(
            x=shelf_x, y=shelf_y,
            mode='markers',
            marker=dict(size=18, color='lightgray', symbol='square', line=dict(color='black', width=1)),
            text=[f"Shelf {i}" for i in range(len(shelf_x))],
            hoverinfo='text',
            name='Shelves'
        ))

        # Plot pallets
        # Plot pallets by priority
        priorities = [p['priority'] for p in pallets if not p['picked_up'] and not p['delivered']]
        unique_priorities = sorted(list(set(priorities)))
        
        priority_colors = {1: 'red', 2: 'orange', 3: 'blue'}
        priority_names = {1: 'High Priority', 2: 'Medium Priority', 3: 'Low Priority'}

        for prio in unique_priorities:
            prio_pallets = [p for p in pallets if p['priority'] == prio and not p['picked_up'] and not p['delivered']]
            
            p_x = [p['position'][0] for p in prio_pallets]
            p_y = [p['position'][1] for p in prio_pallets]
            
            color = priority_colors.get(prio, 'gray')
            size = 14 if prio == 1 else (12 if prio == 2 else 10)
            
            fig.add_trace(go.Scatter(
                x=p_x, y=p_y,
                mode='markers',
                marker=dict(size=size, color=color, symbol='diamond', line=dict(color='black', width=1)),
                text=[f"Pallet {p['id']}<br>Priority: {p['priority']}<br>Dest: {p['destination']}" for p in prio_pallets],
                hoverinfo='text',
                name=f"{priority_names.get(prio, f'Priority {prio}')}"
            ))

            # Draw destinations for these pallets
            d_x = [p['destination'][0] for p in prio_pallets]
            d_y = [p['destination'][1] for p in prio_pallets]
            
            fig.add_trace(go.Scatter(
                x=d_x, y=d_y,
                mode='markers',
                marker=dict(size=8, color='lightgreen', symbol='x'),
                text=[f"Destination for Pallet {p['id']}" for p in prio_pallets],
                hoverinfo='text',
                name='Destinations',
                showlegend=False,
                opacity=0.7
            ))

        # Plot assignment lines
        for pallet in pallets:
            if not pallet['picked_up'] and not pallet['delivered'] and pallet.get('assigned_lgv') is not None:
                assigned_id = pallet['assigned_lgv']
                lgv = next((l for l in lgvs if l['id'] == assigned_id), None)
                if lgv:
                    fig.add_trace(go.Scatter(
                        x=[lgv['position'][0], pallet['position'][0]],
                        y=[lgv['position'][1], pallet['position'][1]],
                        mode='lines',
                        line=dict(color='gray', width=1, dash='dot'),
                        showlegend=False,
                        opacity=0.5
                    ))

        # Plot LGVs
        colors = px.colors.qualitative.Plotly
        for i, lgv in enumerate(lgvs):
            color = colors[i % len(colors)]

            # LGV position
            fig.add_trace(go.Scatter(
                x=[lgv['position'][0]],
                y=[lgv['position'][1]],
                mode='markers+text',
                marker=dict(size=20, color=color, symbol='circle'),
                text=[f"LGV{lgv['id']}"],
                textposition="top center",
                name=f"LGV {lgv['id']}"
            ))

            # Direction indicator
            direction = lgv['direction']
            dx = np.cos(direction) * 0.5
            dy = np.sin(direction) * 0.5

            fig.add_annotation(
                x=lgv['position'][0] + dx,
                y=lgv['position'][1] + dy,
                ax=lgv['position'][0],
                ay=lgv['position'][1],
                xref='x', yref='y',
                axref='x', ayref='y',
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor=color
            )

        # Update layout
        height, width = grid.shape
        fig.update_layout(
            title="Warehouse Layout",
            xaxis=dict(range=[0, height], title="X"),
            yaxis=dict(range=[0, width], title="Y"),
            height=600,
            width=800,
            showlegend=True
        )

        return fig

    @staticmethod
    def plot_metrics_over_time(results: Dict):
        """Plot various metrics over episodes"""
        episode_rewards = results.get('episode_rewards', [])
        episode_lengths = results.get('episode_lengths', [])

        if not episode_rewards:
            return None

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Cumulative Reward', 'Episode Length', 'Success Rate', 'Moving Average Reward'),
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )

        episodes = list(range(len(episode_rewards)))

        # Cumulative reward
        cumulative_rewards = np.cumsum(episode_rewards)
        fig.add_trace(
            go.Scatter(x=episodes, y=cumulative_rewards,
                      mode='lines', name='Cumulative Reward'),
            row=1, col=1
        )

        # Episode length
        fig.add_trace(
            go.Scatter(x=episodes, y=episode_lengths,
                      mode='lines', name='Episode Length'),
            row=1, col=2
        )

        # Success rate (moving average)
        window = 20
        if len(episode_rewards) >= window:
            success_threshold = np.median(episode_rewards)
            successes = [1 if r > success_threshold else 0 for r in episode_rewards]
            success_rate = np.convolve(successes, np.ones(window) / window, mode='valid')
            fig.add_trace(
                go.Scatter(x=list(range(len(success_rate))), y=success_rate,
                          mode='lines', name='Success Rate'),
                row=2, col=1
            )

        # Moving average reward
        if len(episode_rewards) >= window:
            ma_reward = np.convolve(episode_rewards, np.ones(window) / window, mode='valid')
            fig.add_trace(
                go.Scatter(x=list(range(len(ma_reward))), y=ma_reward,
                          mode='lines', name='MA Reward'),
                row=2, col=2
            )

        fig.update_layout(
            title_text="Training Metrics Over Time",
            height=700,
            showlegend=False
        )

        return fig

    @staticmethod
    def plot_heatmap(data: np.ndarray, title: str = "Heatmap"):
        """Plot heatmap"""
        fig = go.Figure(data=go.Heatmap(
            z=data,
            colorscale='Viridis'
        ))

        fig.update_layout(
            title=title,
            height=500,
            width=700
        )

        return fig

    @staticmethod
    def plot_lgv_statistics(env_state: Dict):
        """Plot individual LGV statistics"""
        lgvs = env_state['lgvs']

        if not lgvs:
            return None

        # Extract statistics
        lgv_ids = [lgv['id'] for lgv in lgvs]
        distances = [lgv['total_distance'] for lgv in lgvs]
        deliveries = [lgv['total_deliveries'] for lgv in lgvs]
        collisions = [lgv['collision_count'] for lgv in lgvs]

        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Total Distance', 'Deliveries', 'Collisions'),
            horizontal_spacing=0.12
        )

        # Distance
        fig.add_trace(
            go.Bar(x=lgv_ids, y=distances, name='Distance', marker_color='blue'),
            row=1, col=1
        )

        # Deliveries
        fig.add_trace(
            go.Bar(x=lgv_ids, y=deliveries, name='Deliveries', marker_color='green'),
            row=1, col=2
        )

        # Collisions
        fig.add_trace(
            go.Bar(x=lgv_ids, y=collisions, name='Collisions', marker_color='red'),
            row=1, col=3
        )

        fig.update_layout(
            title_text="LGV Performance Statistics",
            height=400,
            showlegend=False
        )

        return fig

    @staticmethod
    def plot_radar_comparison(comparison_data: Dict):
        """Create radar chart comparing algorithms"""
        categories = ['Mean Reward', 'Completion Rate', 'Efficiency', 'Low Collisions', 'Speed']

        fig = go.Figure()

        for algo_name, metrics in comparison_data.items():
            # Normalize metrics to 0-1 scale
            values = [
                metrics.get('mean_reward', 0) / 100,
                metrics.get('mean_completion_rate', 0),
                metrics.get('efficiency_score', 0) / 100,
                1 - min(metrics.get('mean_collisions', 10) / 10, 1),
                1 - min(metrics.get('mean_episode_length', 500) / 500, 1)
            ]

            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=algo_name
            ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Algorithm Performance Radar Chart",
            height=500
        )

        return fig
