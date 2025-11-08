import numpy as np
import matplotlib.pyplot as plt
import os

def load_csv(filename):
    """Load CSV file into numpy array"""
    return np.genfromtxt(filename, delimiter=',')

def plot_heatmap(data, step, output_dir="plots"):
    """Create and save a heatmap plot"""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(data, cmap='hot', origin='lower', extent=[0, 1, 0, 1])
    plt.colorbar(label='Temperature')
    plt.title(f'2D Heat Distribution - Step {step}')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    output_file = f"{output_dir}/step_{step:04d}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_file}")

def create_animation():
    """Create animation from saved plots"""
    import glob
    from matplotlib.animation import FuncAnimation
    
    # Find all plot files
    plot_files = sorted(glob.glob("plots/step_*.png"))
    
    if not plot_files:
        print("No plot files found. Run the simulation first.")
        return
    
    # Create animation (this is a simple version)
    fig, ax = plt.subplots(figsize=(10, 8))
    
    def update(frame):
        ax.clear()
        img = plt.imread(plot_files[frame])
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f'Step {frame * 100}')  # Adjust based on your save interval
    
    anim = FuncAnimation(fig, update, frames=len(plot_files), interval=200)
    anim.save('plots/heat_evolution.gif', writer='pillow', fps=5)
    print("Animation saved as plots/heat_evolution.gif")

if __name__ == "__main__":
    # Create plots directory
    os.makedirs("plots", exist_ok=True)
    
    # Plot initial condition
    try:
        initial_data = load_csv("results/initial.csv")
        plot_heatmap(initial_data, "initial")
        
        # Plot final result
        final_data = load_csv("results/final.csv")
        plot_heatmap(final_data, "final")
        
        print("Plots created successfully!")
        
        # Uncomment to create animation (requires all intermediate steps)
        # create_animation()
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run the C++ simulation first to generate results.")
