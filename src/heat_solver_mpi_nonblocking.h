#ifndef HEAT_SOLVER_MPI_H
#define HEAT_SOLVER_MPI_H

#include <vector>
#include <mpi.h>

struct DistributedGrid {
    int global_nx, global_ny;
    int local_nx, local_ny;
    int allocated_nx, allocated_ny;
    int start_x, start_y;
    int px, py;
    std::vector<std::vector<double>> local_data;
    int north, south, east, west;
    
    DistributedGrid(int global_nx, int global_ny, int grid_px, int grid_py, int px, int py);
    void print_info(int rank) const;
};

class MPIHeatSolver2D {
private:
    DistributedGrid& grid;
    double dx, dy, dt, alpha;
    double left_bc, right_bc, top_bc, bottom_bc;
    bool use_dirichlet;
    MPI_Comm comm;
    int rank;
    std::vector<std::vector<double>> T_temp;

public:
    MPIHeatSolver2D(DistributedGrid& grid, double Lx, double Ly, double alpha,
                   double left_bc, double right_bc, double top_bc, double bottom_bc,
                   bool use_dirichlet, MPI_Comm comm);
    
    void initialize_gaussian(double center_x, double center_y, double amplitude, double spread);
    
    // Communication methods
    void exchange_halos_blocking();
    void exchange_halos_nonblocking();
    
    // Simulation steps
    void step();
    void step_nonblocking();
    
    void apply_boundary_conditions();
    double calculate_max_timestep() const;
    const DistributedGrid& get_grid() const { return grid; }
};

#endif

