import matplotlib.pyplot as plt

# ================================
# STRONG SCALING (your real data)
# ================================
strong_procs = [1, 2, 4, 8, 16]
strong_times = [17, 12, 13, 37, 56]  # ms

# Compute speedup
speedup = [strong_times[0] / t for t in strong_times]

plt.figure(figsize=(8,6))
plt.plot(strong_procs, speedup, marker='o', linewidth=2)
plt.title("Strong Scaling Speedup (100x100 Grid)", fontsize=14)
plt.xlabel("Number of Processes", fontsize=12)
plt.ylabel("Speedup", fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(strong_procs)
plt.savefig("strong_scaling.png", dpi=150)
plt.close()
print("Saved strong_scaling.png")


# ================================
# WEAK SCALING (your real data)
# ================================
weak_procs = [1, 4, 16]
weak_times = [82, 164, 937]  # ms

plt.figure(figsize=(8,6))
plt.plot(weak_procs, weak_times, marker='o', linewidth=2, color="orange")
plt.title("Weak Scaling Runtime (200x200 per Process)", fontsize=14)
plt.xlabel("Number of Processes", fontsize=12)
plt.ylabel("Runtime (ms)", fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(weak_procs)
plt.savefig("weak_scaling.png", dpi=150)
plt.close()
print("Saved weak_scaling.png")

print("\nAll scaling plots generated successfully!")

