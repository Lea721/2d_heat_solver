#include "heat_solver_mpi.h"
#include <cmath>
#include <iostream>
#include <algorithm>

DistributedGrid::DistributedGrid(int global_nx, int global_ny, 
                               int grid_px, int grid_py, int px, int py)
    : global_nx(global_nx), global_ny(global_ny), px(px), py(py) {
    
    // Calculate local domain decomposition
    local_nx = global_nx / grid_px;
    local_ny = global_ny / grid_py;
    
    // Handle remainder for last process in each dimension
    if (px == grid_px - 1) local_nx = global_nx - (grid_px - 1) * local_nx;
    if (py == grid_py - 1) local_ny = global_ny - (grid_py - 1) * local_ny;
    
    // Calculate starting indices
    start_x = (global_nx / grid_px) * px;
    start_y = (global_ny / grid_py) * py;
    
    // Allocate with halo regions (2 cells on each side)
    allocated_nx = local_nx + 2;
    allocated_ny = local_ny + 2;
    local_data.resize(allocated_ny, std::vector<double>(allocated_nx, 0.0));
    
    // Initialize neighbors (will be set by MPI_Cart_shift)
    north = south = east = west = MPI_PROC_NULL;
}

void DistributedGrid::print_info(int rank) const {
    std::cout << "Rank " << rank << " (" << px << "," << py << "): " 
              << "local=" << local_nx << "x" << local_ny
              << ", start=(" << start_x << "," << start_y << ")"
              << ", allocated=" << allocated_nx << "x" << allocated_ny 
              << std::endl;
}

MPIHeatSolver2D::MPIHeatSolver2D(DistributedGrid& grid, double Lx, double Ly, double alpha,
                               double left_bc, double right_bc, double top_bc, double bottom_bc,
                               bool use_dirichlet, MPI_Comm comm)
    : grid(grid), alpha(alpha), use_dirichlet(use_dirichlet),
      left_bc(left_bc), right_bc(right_bc), top_bc(top_bc), bottom_bc(bottom_bc),
      comm(comm) {
    
    MPI_Comm_rank(comm, &rank);
    
    // Calculate grid spacing
    dx = Lx / (grid.global_nx - 1);
    dy = Ly / (grid.global_ny - 1);
    
    // Calculate stable time step
    dt = 0.25 * calculate_max_timestep();
    
    // Initialize temporary field
    T_temp.resize(grid.allocated_ny, std::vector<double>(grid.allocated_nx, 0.0));
    
    // Set up neighbor communication using Cartesian topology
    MPI_Cart_shift(comm, 0, 1, &grid.west, &grid.east);  // x-dimension
    MPI_Cart_shift(comm, 1, 1, &grid.south, &grid.north); // y-dimension
    
    if (rank == 0) {
        std::cout << "MPI Solver initialized" << std::endl;
        std::cout << "Grid spacing: dx=" << dx << ", dy=" << dy << std::endl;
        std::cout << "Time step: " << dt << std::endl;
    }
}

double MPIHeatSolver2D::calculate_max_timestep() const {
    return std::min(dx*dx, dy*dy) / (4.0 * alpha);
}

void MPIHeatSolver2D::initialize_gaussian(double center_x, double center_y, 
                                        double amplitude, double spread) {
    for (int i = 0; i < grid.allocated_ny; ++i) {
        // Convert local index to global y coordinate
        double y = (grid.start_y + i - 1) * dy; // -1 because of halo offset
        
        for (int j = 0; j < grid.allocated_nx; ++j) {
            // Convert local index to global x coordinate
            double x = (grid.start_x + j - 1) * dx; // -1 because of halo offset
            
            // Only initialize interior points (not halo regions)
            if (i >= 1 && i < grid.allocated_ny - 1 && 
                j >= 1 && j < grid.allocated_nx - 1) {
                double r2 = (x - center_x)*(x - center_x) + (y - center_y)*(y - center_y);
                grid.local_data[i][j] = amplitude * exp(-r2 / (2.0 * spread * spread));
            }
        }
    }
    
    // Initialize temporary field
    T_temp = grid.local_data;
}

void MPIHeatSolver2D::exchange_halos_blocking() {
    MPI_Status status;
    
    // Exchange in x-direction (left/right)
    if (grid.east != MPI_PROC_NULL) {
        // Send to east, receive from west
        std::vector<double> send_buffer(grid.local_ny);
        std::vector<double> recv_buffer(grid.local_ny);
        
        // Prepare send buffer (right boundary)
        for (int i = 0; i < grid.local_ny; ++i) {
            send_buffer[i] = grid.local_data[i + 1][grid.local_nx];
        }
        
        MPI_Sendrecv(send_buffer.data(), grid.local_ny, MPI_DOUBLE, grid.east, 0,
                    recv_buffer.data(), grid.local_ny, MPI_DOUBLE, grid.west, 0,
                    comm, &status);
        
        // Unpack receive buffer (left halo)
        for (int i = 0; i < grid.local_ny; ++i) {
            grid.local_data[i + 1][0] = recv_buffer[i];
        }
    }
    
    if (grid.west != MPI_PROC_NULL) {
        // Send to west, receive from east
        std::vector<double> send_buffer(grid.local_ny);
        std::vector<double> recv_buffer(grid.local_ny);
        
        // Prepare send buffer (left boundary)
        for (int i = 0; i < grid.local_ny; ++i) {
            send_buffer[i] = grid.local_data[i + 1][1];
        }
        
        MPI_Sendrecv(send_buffer.data(), grid.local_ny, MPI_DOUBLE, grid.west, 1,
                    recv_buffer.data(), grid.local_ny, MPI_DOUBLE, grid.east, 1,
                    comm, &status);
        
        // Unpack receive buffer (right halo)
        for (int i = 0; i < grid.local_ny; ++i) {
            grid.local_data[i + 1][grid.local_nx + 1] = recv_buffer[i];
        }
    }
    
    // Exchange in y-direction (top/bottom)
    if (grid.north != MPI_PROC_NULL) {
        // Send to north, receive from south
        std::vector<double> send_buffer(grid.local_nx);
        std::vector<double> recv_buffer(grid.local_nx);
        
        // Prepare send buffer (top boundary)
        for (int j = 0; j < grid.local_nx; ++j) {
            send_buffer[j] = grid.local_data[grid.local_ny][j + 1];
        }
        
        MPI_Sendrecv(send_buffer.data(), grid.local_nx, MPI_DOUBLE, grid.north, 2,
                    recv_buffer.data(), grid.local_nx, MPI_DOUBLE, grid.south, 2,
                    comm, &status);
        
        // Unpack receive buffer (bottom halo)
        for (int j = 0; j < grid.local_nx; ++j) {
            grid.local_data[0][j + 1] = recv_buffer[j];
        }
    }
    
    if (grid.south != MPI_PROC_NULL) {
        // Send to south, receive from north
        std::vector<double> send_buffer(grid.local_nx);
        std::vector<double> recv_buffer(grid.local_nx);
        
        // Prepare send buffer (bottom boundary)
        for (int j = 0; j < grid.local_nx; ++j) {
            send_buffer[j] = grid.local_data[1][j + 1];
        }
        
        MPI_Sendrecv(send_buffer.data(), grid.local_nx, MPI_DOUBLE, grid.south, 3,
                    recv_buffer.data(), grid.local_nx, MPI_DOUBLE, grid.north, 3,
                    comm, &status);
        
        // Unpack receive buffer (top halo)
        for (int j = 0; j < grid.local_nx; ++j) {
            grid.local_data[grid.local_ny + 1][j + 1] = recv_buffer[j];
        }
    }
}

void MPIHeatSolver2D::apply_boundary_conditions() {
    // Apply physical boundary conditions only if this process is on the global boundary
    
    // Left boundary
    if (grid.start_x == 0 && use_dirichlet) {
        for (int i = 1; i <= grid.local_ny; ++i) {
            grid.local_data[i][1] = left_bc; // Left interior boundary
        }
    }
    
    // Right boundary
    if (grid.start_x + grid.local_nx == grid.global_nx && use_dirichlet) {
        for (int i = 1; i <= grid.local_ny; ++i) {
            grid.local_data[i][grid.local_nx] = right_bc; // Right interior boundary
        }
    }
    
    // Bottom boundary
    if (grid.start_y == 0 && use_dirichlet) {
        for (int j = 1; j <= grid.local_nx; ++j) {
            grid.local_data[1][j] = bottom_bc; // Bottom interior boundary
        }
    }
    
    // Top boundary
    if (grid.start_y + grid.local_ny == grid.global_ny && use_dirichlet) {
        for (int j = 1; j <= grid.local_nx; ++j) {
            grid.local_data[grid.local_ny][j] = top_bc; // Top interior boundary
        }
    }
}

void MPIHeatSolver2D::step() {
    // Exchange halo regions with neighbors
    exchange_halos_blocking();
    
    // Apply physical boundary conditions to interior points
    apply_boundary_conditions();
    
    // Update interior points using explicit finite difference
    for (int i = 1; i <= grid.local_ny; ++i) {
        for (int j = 1; j <= grid.local_nx; ++j) {
            double d2T_dx2 = (grid.local_data[i][j-1] - 2.0 * grid.local_data[i][j] + grid.local_data[i][j+1]) / (dx * dx);
            double d2T_dy2 = (grid.local_data[i-1][j] - 2.0 * grid.local_data[i][j] + grid.local_data[i+1][j]) / (dy * dy);
            
            T_temp[i][j] = grid.local_data[i][j] + alpha * dt * (d2T_dx2 + d2T_dy2);
        }
    }
    
    // Swap fields
    std::swap(grid.local_data, T_temp);
}
