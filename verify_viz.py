
import numpy as np
import yaml
from src.environment.warehouse_env import WarehouseEnv
from src.visualization.animator import WarehouseAnimator
from src.visualization.plotter import Plotter

print("Initializing WarehouseEnv...")
env = WarehouseEnv('config.yaml')
obs, info = env.reset()

print("Checking environment state structure...")
state = env.get_state()
if 'pallets' in state and len(state['pallets']) > 0:
    first_pallet = state['pallets'][0]
    print(f"Pallet Keys: {first_pallet.keys()}")
    if 'priority' not in first_pallet:
        print("ERROR: 'priority' missing from pallet state!")
    if 'assigned_lgv' not in first_pallet:
        print("ERROR: 'assigned_lgv' missing from pallet state!")
else:
    print("WARNING: No pallets found in initial state.")

print("Testing Animator...")
animator = WarehouseAnimator(env.width, env.height)
animator.add_frame(state)
try:
    fig = animator.create_plotly_animation()
    print("Animator: Plotly animation created successfully.")
    # Check for hovertext
    if fig.frames and 'data' in fig.frames[0]:
        traces = fig.frames[0].data
        shelf_trace = next((t for t in traces if t.name == 'Shelves'), None)
        if shelf_trace and getattr(shelf_trace, 'hovertext', None):
            print("Animator: Shelf hovertext found.")
        else:
            print("WARNING: Animator Shelf hovertext MISSING.")
except Exception as e:
    print(f"Animator ERROR: {e}")

print("Testing Plotter...")
try:
    fig = Plotter.plot_warehouse_layout(state)
    print("Plotter: Layout plot created successfully.")
    # Check for hovertext
    if fig.data:
        shelf_trace = next((t for t in fig.data if t.name == 'Shelves'), None)
        if shelf_trace and getattr(shelf_trace, 'text', None): # Static plot uses 'text' or 'hovertext' depending on mode
             print("Plotter: Shelf hovertext found.")
        else:
             print("WARNING: Plotter Shelf hovertext MISSING.")
except Exception as e:
    print(f"Plotter ERROR: {e}")

print("Verification complete.")
