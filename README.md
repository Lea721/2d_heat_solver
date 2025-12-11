```markdown
# 2D Heat Solver
**Project:** Parallel 2D Heat Equation Solver with MPI and OpenGL Visualization  
**Description:**  
This project implements a 2D heat equation solver supporting serial execution, MPI parallelism, non-blocking MPI communication, and OpenGL visualization. It includes scripts for building, testing, and plotting results, as well as strong and weak scaling experiments.

---

## Table of Contents
1. [Prerequisites](#prerequisites)  
2. [Project Structure](#project-structure)  
3. [Building and Running](#building-and-running)  
   - [Serial Version](#serial-version)  
   - [MPI Parallel Version](#mpi-parallel-version)  
   - [MPI Non-Blocking Version](#mpi-non-blocking-version)  
   - [Dirichlet vs Neumann Analysis](#dirichlet-vs-neumann-analysis)  
   - [OpenGL Visualization](#opengl-visualization)  
4. [Running MPI Across Multiple PCs](#running-mpi-across-multiple-pcs)  
5. [Scaling Experiments](#scaling-experiments)  
6. [Tips](#tips)  

---

## Prerequisites
- Linux or macOS environment  
- C++11 or C++17 compiler (`g++`)  
- MPI library (OpenMPI recommended)  
- Python 3.x with `matplotlib` and `numpy`  
- OpenGL libraries (`GL`, `GLU`, `glut`) for visualization  

---

## Project Structure
```

2d_heat_solver/
├── src/                    # Source code
├── build/                  # Compiled binaries
├── scripts/                # Build and test scripts
├── visualization/          # Python scripts for plotting
├── output/                 # Output files for scaling experiments

````

---

## Building and Running

### Serial Version
```bash
cd ~/2d_heat_solver
chmod +x scripts/build.sh
./scripts/build.sh
./build/heat_solver_serial

cd ~/2d_heat_solver/visualization
python3 plot_results.py
````

---

### MPI Parallel Version

```bash
cd ~/2d_heat_solver
chmod +x scripts/*.sh
./scripts/build_mpi.sh
./scripts/test_parallel.sh

cd ~/2d_heat_solver/visualization
python3 plot_results_mpi.py
```

---

### MPI Non-Blocking Version

```bash
cd ~/2d_heat_solver
chmod +x scripts/*.sh
./scripts/build_mpi_nonblocking.sh
./scripts/test_parallel_nonblocking.sh

cd ~/2d_heat_solver/visualization
python3 plot_results_mpi_nonblocking.py
```

---

### Dirichlet vs Neumann Analysis

```bash
mpic++ -std=c++11 -O2 src/dirichletVSneuman.cpp src/heat_solver_mpi_nonblocking.cpp -o build/dirichletVSneuman

mpirun -np 4 ./build/dirichletVSneuman dirichlet
mpirun -np 4 ./build/dirichletVSneuman neumann

cd visualization
python3 analysis.py
python3 visual_check.py
```

---

### OpenGL Visualization

```bash
mpic++ -std=c++17 \
    src/heat_solver_mpi_visual.cpp \
    src/heat_solver_mpi_nonblocking.cpp \
    visualizer/visualizer_opengl.cpp \
    -lglut -lGLU -lGL \
    -o build/heat_vis

mpirun --oversubscribe -np 4 ./build/heat_vis
```

---

## Running MPI Across Multiple PCs

1. **Set up bridged network:** Ensure all VMs are on the same network (`ip a`) and can ping each other.

   ```bash
   ping 10.100.100.101
   ```
2. **Test SSH connectivity:**

   ```bash
   ssh progparallele@10.100.100.101
   ```
3. **(Optional) Enable passwordless SSH:**

   ```bash
   ssh-keygen -t rsa   # Press Enter for defaults, leave passphrase empty
   ssh-copy-id progparallele@10.100.100.101
   ssh progparallele@10.100.100.101
   ```
4. **Prepare MPI hosts file:**
   Example `~/hosts`:

   ```
   10.100.100.100 slots=2
   10.100.100.101 slots=2
   ```
5. **Copy project to all VMs:**

   ```bash
   scp -r ~/Desktop/2d_heat_solver/* progparallele@10.100.100.101:~/Desktop/2d_heat_solver/
   scp -r build/ progparallele@10.100.100.101:~/Desktop/2d_heat_solver/
   ```
6. **Run MPI program:**

   ```bash
   mpirun -np 4 --hostfile ~/hosts -x DISPLAY= -x XAUTHORITY= ~/Desktop/2d_heat_solver/build/heat_solver_mpi
   ```

---

## Scaling Experiments

### Strong Scaling

```bash
mpirun --oversubscribe -np 1 ./build/dirichletVSneuman dirichlet > output/strong_1.txt
mpirun --oversubscribe -np 2 ./build/dirichletVSneuman dirichlet > output/strong_2.txt
mpirun --oversubscribe -np 4 ./build/dirichletVSneuman dirichlet > output/strong_4.txt
mpirun --oversubscribe -np 8 ./build/dirichletVSneuman dirichlet > output/strong_8.txt
mpirun --oversubscribe -np 16 ./build/dirichletVSneuman dirichlet > output/strong_16.txt
```

### Weak Scaling

1. Set `nx = ny = 200` → recompile and run:

```bash
mpic++ -std=c++11 -O2 src/dirichletVSneuman.cpp src/heat_solver_mpi_nonblocking.cpp -o build/dirichletVSneuman
mpirun --oversubscribe -np 1 ./build/dirichletVSneuman dirichlet > output/weak_1.txt
```

2. Set `nx = ny = 400` → recompile → run with 4 processes:

```bash
mpirun --oversubscribe -np 4 ./build/dirichletVSneuman dirichlet > output/weak_4.txt
```

3. Set `nx = ny = 800` → recompile → run with 16 processes:

```bash
mpirun --oversubscribe -np 16 ./build/dirichletVSneuman dirichlet > output/weak_16.txt
```

---

## Tips

* Ensure all VMs have identical folder structures.
* Test SSH connectivity from master VM to all slave VMs.
* Use `hostname` in MPI code to confirm process distribution.
* For OpenGL, pass `-x DISPLAY= -x XAUTHORITY=` when running MPI on multiple nodes to prevent GUI issues.

---
Do you want me to do that?
```
