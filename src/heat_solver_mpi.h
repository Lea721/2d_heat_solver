#ifndef HEAT_SOLVER_MPI_H
#define HEAT_SOLVER_MPI_H

#include <vector>
#include <mpi.h>

struct DistributedGrid {
    // Global grid dimensions
    int global_nx, global_ny;
    
    // Local grid dimensions (excluding halos)
    int local_nx, local_ny;
    
    // Local grid dimensions (including halos)
    int allocated_nx, allocated_ny;
    
    // Starting indices in global grid
    int start_x, start_y;
    
    // Process grid coordinates
    int px, py;
    
    // Local data with halo regions
    std::vector<std::vector<double>> local_data;
    
    // Neighbor ranks
    int north, south, east, west;
    
    DistributedGrid(int global_nx, int global_ny, int grid_px, int grid_py, int px, int py);
    
    void print_info(int rank) const;
};

class MPIHeatSolver2D {
private:
    DistributedGrid& grid;
    
    // Grid parameters
    double dx, dy;
    double dt;
    double alpha;
    
    // Boundary conditions
    double left_bc, right_bc, top_bc, bottom_bc;
    bool use_dirichlet;
    
    // MPI
    MPI_Comm comm;
    int rank;
    
    // Temporary field for updates
    std::vector<std::vector<double>> T_temp;
    
public:
    MPIHeatSolver2D(DistributedGrid& grid, double Lx, double Ly, double alpha,
                   double left_bc, double right_bc, double top_bc, double bottom_bc,
                   bool use_dirichlet, MPI_Comm comm);
    
    // Initialize temperature field
    void initialize_gaussian(double center_x, double center_y, double amplitude, double spread);
    void initialize_uniform(double temperature);
    
    // Communication
    void exchange_halos();
    void exchange_halos_blocking();  // Simple blocking version first
    
    // Simulation step
    void step();
    
    // Boundary condition application
    void apply_boundary_conditions();
    
    // Utility
    double calculate_max_timestep() const;
    bool is_boundary_process() const;
};

#endif
