import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection

def visualize_na_atrium_architecture():
    # --- 1. Define the architecture parameters ---
    grid_size = 7      # 7x7 layer grid
    num_layers = 6     # 5 layers for storage/computation + 1 layer for readout
    spacing_xy = 5.0   # um
    spacing_z = 25.0   # um (typical high PSF axial spacing)
    
    # Define the center computation zone range (indices 2, 3, 4)
    center_range = range(2, 5)
    rydberg_radius = 12.0 # um

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(projection='3d')

    # Generate grid coordinates
    x, y, z = np.meshgrid(np.arange(grid_size) * spacing_xy,
                          np.arange(grid_size) * spacing_xy,
                          np.arange(num_layers) * spacing_z)
    
    # Flatten the coordinates
    x_flat, y_flat, z_flat = x.flatten(), y.flatten(), z.flatten()
    grid_coords = list(zip(x_flat, y_flat, z_flat))

    # --- 2. Physical partition logic (Compiler logical abstraction) ---
    atom_colors = []
    atom_sizes = []
    
    # Track atoms at risk (on the Skewer path)
    at_risk_atoms = []
    
    # Select a target to execute Gate (e.g. center of Layer 3 (3, 3, 2))
    target_ix, target_iy, target_iz = 3, 3, 2
    tx, ty, tz = target_ix * spacing_xy, target_iy * spacing_xy, target_iz * spacing_z

    for xi, yi, zi in grid_coords:
        ix, iy, iz = int(xi/spacing_xy), int(yi/spacing_xy), int(zi/spacing_z)
        
        # Top layer readout zone
        if iz == num_layers - 1:
            atom_colors.append('lightgreen')
            atom_sizes.append(30)
        # Center computation zone (Layers 1-5)
        elif ix in center_range and iy in center_range:
            # Mark the selected target atom
            if ix == target_ix and iy == target_iy and iz == target_iz:
                atom_colors.append('red')
                atom_sizes.append(120) # Activated state large point
            else:
                atom_colors.append('tomato') # Potential computation position
                atom_sizes.append(40)
                # Check if it is on the vertical interference column of the target
                if ix == target_ix and iy == target_iy:
                    at_risk_atoms.append((xi, yi, zi))
        # Outer storage zone (Layers 1-5)
        else:
            atom_colors.append('royalblue') # Stable storage
            atom_sizes.append(40)

    # --- 3. Draw 3D scatter plot ---
    sc = ax.scatter(x_flat, y_flat, z_flat, c=atom_colors, s=atom_sizes, alpha=0.8)

    # --- 4. 可视化 Compiler 物理约束 ---
    
    # A. Visualize The Skewer (vertical penetrating laser column)
    z_line = np.linspace(0, (num_layers - 2) * spacing_z, 100)
    ax.plot([tx]*100, [ty]*100, z_line, 'r--', linewidth=2.5, alpha=0.9, label='Laser Skewer (Addressing Beam)')

    # B. Visualize threatened atoms (atoms on non-target layers on the Skewer path)
    if at_risk_atoms:
        risky_x, risky_y, risky_z = zip(*at_risk_atoms)
        ax.scatter(risky_x, risky_y, risky_z, edgecolors='black', facecolors='none', s=100, linewidths=2, label='Crosstalk-Risk Atoms')

    # C. Visualize Rydberg Blockade Sphere (omnidirectional entanglement range)
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    sphere_x = tx + rydberg_radius * np.cos(u) * np.sin(v)
    sphere_y = ty + rydberg_radius * np.sin(u) * np.sin(v)
    sphere_z = tz + rydberg_radius * np.cos(v)
    ax.plot_wireframe(sphere_x, sphere_y, sphere_z, color="red", alpha=0.08, label='Rydberg Blockade Sphere')

    # --- 5. Beautiful view ---
    # Define partition legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Storage Zone (L1-L5)', markerfacecolor='royalblue', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Interaction Zone (L1-L5)', markerfacecolor='tomato', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Target Active Atom', markerfacecolor='red', markersize=15),
        Line2D([0], [0], marker='o', color='w', label='Readout Plane (L6)', markerfacecolor='lightgreen', markersize=10),
        Line2D([0], [0], color='r', linestyle='--', label='Laser Skewer'),
        Line2D([0], [0], marker='o', color='w', label='Crosstalk Risk', markeredgecolor='black', markerfacecolor='none', markersize=10, markeredgewidth=2),
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))
    
    ax.set_xlabel('X ($\mu m$)')
    ax.set_ylabel('Y ($\mu m$)')
    ax.set_zlabel('Z ($\mu m$)')
    ax.set_title(f'Visualizing Atrium-style 3D NA Architecture\nSelected Gate at Layer 3 Center ({spacing_z} $\mu m$ spacing)')
    
    # Keep equal proportions to prevent Z-axis compression
    fig_scale = [grid_size*spacing_xy, grid_size*spacing_xy, (num_layers)*spacing_z]
    ax.set_box_aspect(fig_scale) 
    
    plt.show()

if __name__ == "__main__":
    visualize_na_atrium_architecture()