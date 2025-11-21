#include "heat_solver_mpi.h"
#include <cmath>
#include <iostream>
#include <algorithm>
#include <iomanip>

DistributedGrid::DistributedGrid(int global_nx, int global_ny, 
                               int grid_px, int grid_py, int px, int py)
    : global_nx(global_nx), global_ny(global_ny), px(px), py(py) {
    
    local_nx = global_nx / grid_px;
    local_ny = global_ny / grid_py;
    
    if (px == grid_px - 1) local_nx = global_nx - (grid_px - 1) * local_nx;
    if (py == grid_py - 1) local_ny = global_ny - (grid_py - 1) * local_ny;
    
    start_x = (global_nx / grid_px) * px;
    start_y = (global_ny / grid_py) * py;
    
    allocated_nx = local_nx + 2;
    allocated_ny = local_ny + 2;
    local_data.resize(allocated_ny, std::vector<double>(allocated_nx, 0.0));
    
    north = south = east = west = MPI_PROC_NULL;
}

void DistributedGrid::print_info(int rank) const {
    std::cout << "Rank " << rank << " (" << px << "," << py << "): " 
              << "local=" << local_nx << "x" << local_ny
              << ", start=(" << start_x << "," << start_y << ")"
              << std::endl;
}

MPIHeatSolver2D::MPIHeatSolver2D(DistributedGrid& grid, double Lx, double Ly, double alpha,
                               double left_bc, double right_bc, double top_bc, double bottom_bc,
                               bool use_dirichlet, MPI_Comm comm)
    : grid(grid), alpha(alpha), use_dirichlet(use_dirichlet),
      left_bc(left_bc), right_bc(right_bc), top_bc(top_bc), bottom_bc(bottom_bc),
      comm(comm) {
    
    MPI_Comm_rank(comm, &rank);
    dx = Lx / (grid.global_nx - 1);
    dy = Ly / (grid.global_ny - 1);
    dt = 0.25 * calculate_max_timestep();
    T_temp.resize(grid.allocated_ny, std::vector<double>(grid.allocated_nx, 0.0));
    
    MPI_Cart_shift(comm, 0, 1, &grid.west, &grid.east);
    MPI_Cart_shift(comm, 1, 1, &grid.south, &grid.north);
    
    if (rank == 0) {
        std::cout << "MPI Solver: dx=" << dx << ", dy=" << dy << ", dt=" << dt << std::endl;
    }
}

double MPIHeatSolver2D::calculate_max_timestep() const {
    return std::min(dx*dx, dy*dy) / (4.0 * alpha);
}

void MPIHeatSolver2D::initialize_gaussian(double center_x, double center_y, 
                                        double amplitude, double spread) {
    for (int i = 1; i <= grid.local_ny; ++i) {
        double y = (grid.start_y + i - 1) * dy;
        for (int j = 1; j <= grid.local_nx; ++j) {
            double x = (grid.start_x + j - 1) * dx;
            double r2 = (x - center_x)*(x - center_x) + (y - center_y)*(y - center_y);
            grid.local_data[i][j] = amplitude * exp(-r2 / (2.0 * spread * spread));
        }
    }
    T_temp = grid.local_data;
}

void MPIHeatSolver2D::exchange_halos_blocking() {
    MPI_Status status;
    
    // Exchange in x-direction
    std::vector<double> send_east(grid.local_ny, 0.0);
    std::vector<double> recv_west(grid.local_ny, 0.0);
    std::vector<double> send_west(grid.local_ny, 0.0);
    std::vector<double> recv_east(grid.local_ny, 0.0);
    
    if (grid.east != MPI_PROC_NULL) {
        for (int i = 0; i < grid.local_ny; ++i) {
            send_east[i] = grid.local_data[i + 1][grid.local_nx];
        }
    }
    if (grid.west != MPI_PROC_NULL) {
        for (int i = 0; i < grid.local_ny; ++i) {
            send_west[i] = grid.local_data[i + 1][1];
        }
    }
    
    MPI_Sendrecv(send_east.data(), grid.local_ny, MPI_DOUBLE, grid.east, 0,
                 recv_west.data(), grid.local_ny, MPI_DOUBLE, grid.west, 0,
                 comm, &status);
    MPI_Sendrecv(send_west.data(), grid.local_ny, MPI_DOUBLE, grid.west, 1,
                 recv_east.data(), grid.local_ny, MPI_DOUBLE, grid.east, 1,
                 comm, &status);
    
    if (grid.west != MPI_PROC_NULL) {
        for (int i = 0; i < grid.local_ny; ++i) {
            grid.local_data[i + 1][0] = recv_west[i];
        }
    }
    if (grid.east != MPI_PROC_NULL) {
        for (int i = 0; i < grid.local_ny; ++i) {
            grid.local_data[i + 1][grid.local_nx + 1] = recv_east[i];
        }
    }
    
    // Exchange in y-direction
    std::vector<double> send_north(grid.local_nx, 0.0);
    std::vector<double> recv_south(grid.local_nx, 0.0);
    std::vector<double> send_south(grid.local_nx, 0.0);
    std::vector<double> recv_north(grid.local_nx, 0.0);
    
    if (grid.north != MPI_PROC_NULL) {
        for (int j = 0; j < grid.local_nx; ++j) {
            send_north[j] = grid.local_data[grid.local_ny][j + 1];
        }
    }
    if (grid.south != MPI_PROC_NULL) {
        for (int j = 0; j < grid.local_nx; ++j) {
            send_south[j] = grid.local_data[1][j + 1];
        }
    }
    
    MPI_Sendrecv(send_north.data(), grid.local_nx, MPI_DOUBLE, grid.north, 2,
                 recv_south.data(), grid.local_nx, MPI_DOUBLE, grid.south, 2,
                 comm, &status);
    MPI_Sendrecv(send_south.data(), grid.local_nx, MPI_DOUBLE, grid.south, 3,
                 recv_north.data(), grid.local_nx, MPI_DOUBLE, grid.north, 3,
                 comm, &status);
    
    if (grid.south != MPI_PROC_NULL) {
        for (int j = 0; j < grid.local_nx; ++j) {
            grid.local_data[0][j + 1] = recv_south[j];
        }
    }
    if (grid.north != MPI_PROC_NULL) {
        for (int j = 0; j < grid.local_nx; ++j) {
            grid.local_data[grid.local_ny + 1][j + 1] = recv_north[j];
        }
    }
}

void MPIHeatSolver2D::apply_boundary_conditions() {
    if (grid.start_x == 0 && use_dirichlet) {
        for (int i = 1; i <= grid.local_ny; ++i) {
            T_temp[i][1] = left_bc;
        }
    }
    if (grid.start_x + grid.local_nx == grid.global_nx && use_dirichlet) {
        for (int i = 1; i <= grid.local_ny; ++i) {
            T_temp[i][grid.local_nx] = right_bc;
        }
    }
    if (grid.start_y == 0 && use_dirichlet) {
        for (int j = 1; j <= grid.local_nx; ++j) {
            T_temp[1][j] = bottom_bc;
        }
    }
    if (grid.start_y + grid.local_ny == grid.global_ny && use_dirichlet) {
        for (int j = 1; j <= grid.local_nx; ++j) {
            T_temp[grid.local_ny][j] = top_bc;
        }
    }
}

void MPIHeatSolver2D::step() {
    exchange_halos_blocking();
    
    for (int i = 1; i <= grid.local_ny; ++i) {
        for (int j = 1; j <= grid.local_nx; ++j) {
            double d2T_dx2 = (grid.local_data[i][j-1] - 2.0 * grid.local_data[i][j] + grid.local_data[i][j+1]) / (dx * dx);
            double d2T_dy2 = (grid.local_data[i-1][j] - 2.0 * grid.local_data[i][j] + grid.local_data[i+1][j]) / (dy * dy);
            T_temp[i][j] = grid.local_data[i][j] + alpha * dt * (d2T_dx2 + d2T_dy2);
        }
    }
    
    apply_boundary_conditions();
    std::swap(grid.local_data, T_temp);
}

