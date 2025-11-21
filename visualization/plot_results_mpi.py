import numpy as np
import matplotlib.pyplot as plt
import os

def load_csv(filename):
    """Load CSV file into numpy array"""
    return np.genfromtxt(filename, delimiter=',')

def plot_heatmap(data, step, output_dir="plots_parallel"):  # ← KEEP THIS
    """Create and save a heatmap plot"""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(data, cmap='hot', origin='lower', vmin=0, vmax=data.max())
    plt.colorbar(label='Temperature')
    plt.title(f'2D Heat Distribution - Step {step}')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    output_file = f"{output_dir}/step_{step}.png"  # ← REMOVE :04d
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_file}")

if __name__ == "__main__":
    # Create plots directory
    os.makedirs("plots", exist_ok=True)  # ← KEEP THIS
    
    # Plot initial condition
    try:
        initial_data = load_csv("../results/initial_parallel.csv")  # ← CHANGED
        plot_heatmap(initial_data, "initial")
        
        # Plot final result
        final_data = load_csv("../results/final_parallel.csv")      # ← CHANGED
        plot_heatmap(final_data, "final")
        
        print("Plots created successfully!")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run the C++ simulation first to generate results.")
