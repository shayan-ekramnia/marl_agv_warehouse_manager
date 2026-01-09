"""
Animation utilities for warehouse simulation
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from typing import List, Dict
import plotly.graph_objects as go


class WarehouseAnimator:
    """Create animations of warehouse operations"""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.frames = []

    def add_frame(self, env_state: Dict):
        """Add a frame to animation"""
        self.frames.append(env_state)

    def create_matplotlib_animation(self, save_path: str = None, fps: int = 10):
        """Create matplotlib animation"""
        fig, ax = plt.subplots(figsize=(12, 10))

        def init():
            ax.clear()
            ax.set_xlim(0, self.width)
            ax.set_ylim(0, self.height)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            ax.set_title('Warehouse LGV Operations')
            return []

        def animate(frame_idx):
            ax.clear()
            ax.set_xlim(0, self.width)
            ax.set_ylim(0, self.height)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)

            if frame_idx >= len(self.frames):
                return []

            frame = self.frames[frame_idx]

            # Draw shelves
            for shelf in frame['shelves']:
                x, y = shelf['position']
                rect = patches.Rectangle(
                    (x - 0.5, y - 0.5), 1, 1,
                    linewidth=1, edgecolor='black',
                    facecolor='gray', alpha=0.5
                )
                ax.add_patch(rect)

            # Draw pallets
            for pallet in frame['pallets']:
                if not pallet['picked_up'] and not pallet['delivered']:
                    x, y = pallet['position']
                    ax.plot(x, y, 'bs', markersize=8, label='Pallet' if frame['pallets'].index(pallet) == 0 else '')

                    # Draw destination
                    dx, dy = pallet['destination']
                    ax.plot(dx, dy, 'g*', markersize=10)
                    ax.plot([x, dx], [y, dy], 'g--', alpha=0.3, linewidth=1)

            # Draw LGVs
            colors = plt.cm.tab10(np.linspace(0, 1, 10))
            for lgv in frame['lgvs']:
                x, y = lgv['position']
                color = colors[lgv['id'] % 10]

                # LGV body
                circle = patches.Circle(
                    (x, y), 0.4,
                    color=color, alpha=0.7
                )
                ax.add_patch(circle)

                # Direction indicator
                direction = lgv['direction']
                dx = np.cos(direction) * 0.6
                dy = np.sin(direction) * 0.6
                ax.arrow(x, y, dx, dy, head_width=0.3, head_length=0.2,
                        fc=color, ec=color, alpha=0.8)

                # Label
                ax.text(x, y + 0.7, f"LGV{lgv['id']}", ha='center',
                       fontsize=8, fontweight='bold')

                # Load indicator
                if lgv['has_load']:
                    ax.text(x, y - 0.7, '📦', ha='center', fontsize=12)

            # Info text
            info = frame.get('info', {})
            info_text = f"Step: {info.get('step', 0)} | "
            info_text += f"Deliveries: {info.get('total_deliveries', 0)} | "
            info_text += f"Completion: {info.get('completion_rate', 0):.1%}"
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            return []

        anim = FuncAnimation(fig, animate, init_func=init,
                           frames=len(self.frames), interval=1000 // fps,
                           blit=True, repeat=True)

        if save_path:
            anim.save(save_path, writer='pillow', fps=fps)

        plt.close()
        return anim

    def create_plotly_animation(self):
        """Create Plotly animation"""
        if not self.frames:
            return None

        # Prepare data for all frames
        frames_data = []

        for frame_idx, frame in enumerate(self.frames):
            frame_traces = []

            # Shelves (static)
            shelf_x = [s['position'][0] for s in frame['shelves']]
            shelf_y = [s['position'][1] for s in frame['shelves']]

            frame_traces.append(go.Scatter(
                x=shelf_x, y=shelf_y,
                mode='markers',
                marker=dict(size=18, color='lightgray', symbol='square', line=dict(color='black', width=1)),
                hovertext=[f"Shelf {i}" for i in range(len(shelf_x))],
                hoverinfo='text',
                name='Shelves'
            ))

            # Pallets
            pallet_x = []
            pallet_y = []
            for pallet in frame['pallets']:
                if not pallet['picked_up'] and not pallet['delivered']:
                    pallet_x.append(pallet['position'][0])
                    pallet_y.append(pallet['position'][1])

            if pallet_x:
                # Group pallets by priority for distinct colors
                priorities = [p['priority'] for p in frame['pallets'] if not p['picked_up'] and not p['delivered']]
                
                # Priority Colors: 1=Red, 2=Orange, 3=Blue
                priority_colors = {
                    1: 'red',
                    2: 'orange', 
                    3: 'blue'
                }
                
                marker_colors = [priority_colors.get(p, 'gray') for p in priorities]
                marker_sizes = [14 if p==1 else (12 if p==2 else 10) for p in priorities]

                priorities_data = [p for p in frame['pallets'] if not p['picked_up'] and not p['delivered']]
                hover_texts = [f"Pallet {p['id']}<br>Priority: {p['priority']}<br>Dest: {p['destination']}" for p in priorities_data]

                frame_traces.append(go.Scatter(
                    x=pallet_x, y=pallet_y,
                    mode='markers',
                    marker=dict(
                        size=marker_sizes, 
                        color=marker_colors, 
                        symbol='diamond',
                        line=dict(color='black', width=1)
                    ),
                    text=hover_texts,
                    hoverinfo='text',
                    name='Pallets (Red=High, Org=Med, Blu=Low)'
                ))

            # Task Assignment Lines
            for pallet in frame['pallets']:
                if not pallet['picked_up'] and not pallet['delivered'] and pallet.get('assigned_lgv') is not None:
                    assigned_lgv_id = pallet['assigned_lgv']
                    # Find LGV position
                    lgv_pos = next((l['position'] for l in frame['lgvs'] if l['id'] == assigned_lgv_id), None)
                    if lgv_pos:
                        frame_traces.append(go.Scatter(
                            x=[lgv_pos[0], pallet['position'][0]],
                            y=[lgv_pos[1], pallet['position'][1]],
                            mode='lines',
                            line=dict(color='gray', width=1, dash='dot'),
                            showlegend=False,
                            opacity=0.5
                        ))

            # LGVs
            lgv_x = [lgv['position'][0] for lgv in frame['lgvs']]
            lgv_y = [lgv['position'][1] for lgv in frame['lgvs']]
            lgv_text = [f"LGV{lgv['id']}" for lgv in frame['lgvs']]

            frame_traces.append(go.Scatter(
                x=lgv_x, y=lgv_y,
                mode='markers+text',
                marker=dict(size=20, color='blue'),
                text=lgv_text,
                textposition="top center",
                name='LGVs'
            ))

            frames_data.append(go.Frame(data=frame_traces, name=str(frame_idx)))

        # Create figure
        fig = go.Figure(
            data=frames_data[0].data if frames_data else [],
            frames=frames_data
        )

        # Add animation controls
        fig.update_layout(
            title="Warehouse Animation",
            xaxis=dict(range=[0, self.width], title="X"),
            yaxis=dict(range=[0, self.height], title="Y"),
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {'label': 'Play', 'method': 'animate',
                     'args': [None, {'frame': {'duration': 100, 'redraw': True},
                                   'fromcurrent': True}]},
                    {'label': 'Pause', 'method': 'animate',
                     'args': [[None], {'frame': {'duration': 0, 'redraw': False},
                                      'mode': 'immediate',
                                      'transition': {'duration': 0}}]}
                ]
            }],
            sliders=[{
                'steps': [
                    {'args': [[f.name], {'frame': {'duration': 0, 'redraw': True},
                                        'mode': 'immediate'}],
                     'label': str(i),
                     'method': 'animate'}
                    for i, f in enumerate(frames_data)
                ],
                'active': 0,
                'y': 0,
                'len': 0.9,
                'x': 0.1,
                'xanchor': 'left',
                'yanchor': 'top'
            }],
            height=700,
            width=900
        )

        return fig

    def clear_frames(self):
        """Clear all frames"""
        self.frames = []

    def get_frame_count(self) -> int:
        """Get number of frames"""
        return len(self.frames)

    @staticmethod
    def create_trajectory_plot(trajectories: Dict[int, List[tuple]]):
        """
        Plot trajectories of all LGVs

        Args:
            trajectories: Dict mapping LGV ID to list of (x, y) positions
        """
        fig = go.Figure()

        colors = ['blue', 'red', 'green', 'orange', 'purple', 'cyan']

        for lgv_id, trajectory in trajectories.items():
            if not trajectory:
                continue

            x_coords = [pos[0] for pos in trajectory]
            y_coords = [pos[1] for pos in trajectory]

            # Plot trajectory
            fig.add_trace(go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='lines+markers',
                name=f'LGV {lgv_id}',
                line=dict(color=colors[lgv_id % len(colors)], width=2),
                marker=dict(size=4)
            ))

            # Mark start and end
            fig.add_trace(go.Scatter(
                x=[x_coords[0]],
                y=[y_coords[0]],
                mode='markers',
                marker=dict(size=15, color=colors[lgv_id % len(colors)],
                          symbol='star'),
                name=f'Start LGV {lgv_id}',
                showlegend=False
            ))

            fig.add_trace(go.Scatter(
                x=[x_coords[-1]],
                y=[y_coords[-1]],
                mode='markers',
                marker=dict(size=15, color=colors[lgv_id % len(colors)],
                          symbol='square'),
                name=f'End LGV {lgv_id}',
                showlegend=False
            ))

        fig.update_layout(
            title="LGV Trajectories",
            xaxis_title="X Position",
            yaxis_title="Y Position",
            height=600,
            width=800,
            showlegend=True
        )

        return fig
