import numpy as np
import matplotlib.pyplot as plt

print("=== FINAL VISUAL INSPECTION ===")

try:
    # Load results
    dirichlet = np.genfromtxt('../results/final_dirichlet.csv', delimiter=',')
    neumann = np.genfromtxt('../results/final_neumann.csv', delimiter=',')
    
    print("Data loaded successfully!")
    print(f"Dirichlet shape: {dirichlet.shape}")
    print(f"Neumann shape: {neumann.shape}")
    
    # Quick statistical check
    print(f"\nDirichlet - Min: {dirichlet.min():.3f}, Max: {dirichlet.max():.3f}, Mean: {dirichlet.mean():.3f}")
    print(f"Neumann   - Min: {neumann.min():.3f}, Max: {neumann.max():.3f}, Mean: {neumann.mean():.3f}")
    
    # Check for physical consistency
    if dirichlet.max() > 50 and neumann.max() > 50:
        print("✅ Temperature ranges are physically reasonable")
    else:
        print("⚠️  Temperature ranges seem low")
    
    # Check that results are different (different BCs should give different results)
    diff_mean = abs(dirichlet.mean() - neumann.mean())
    if diff_mean > 5:  # Arbitrary threshold
        print("✅ Boundary conditions produce significantly different results")
    else:
        print("⚠️  Results are very similar - might indicate issues")
    
    # Create side-by-side comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Dirichlet - should show gradients to boundaries
    im1 = ax1.imshow(dirichlet, cmap='hot', origin='lower')
    ax1.set_title('Dirichlet BCs\n(Heat flows to fixed boundaries)', fontweight='bold')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    plt.colorbar(im1, ax=ax1)
    
    # Neumann - should show more uniform distribution
    im2 = ax2.imshow(neumann, cmap='hot', origin='lower')
    ax2.set_title('Neumann BCs\n(Heat trapped inside - insulated)', fontweight='bold')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig('../results/final_visual_check.png', dpi=150, bbox_inches='tight')
    print("\nVisual inspection plot saved to results/final_visual_check.png")
    
except Exception as e:
    print(f"Error during visual inspection: {e}")

