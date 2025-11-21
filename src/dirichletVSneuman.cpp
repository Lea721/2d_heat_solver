#include "heat_solver_mpi_nonblocking.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <mpi.h>

void save_to_csv_parallel(const DistributedGrid& grid, const std::string& filename, 
                         MPI_Comm comm, int root = 0) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    
    if (rank == root) {
        std::vector<std::vector<double>> global_grid(grid.global_ny, 
                                                   std::vector<double>(grid.global_nx, 0.0));
        
        for (int i = 0; i < grid.local_ny; ++i) {
            for (int j = 0; j < grid.local_nx; ++j) {
                int global_i = grid.start_y + i;
                int global_j = grid.start_x + j;
                global_grid[global_i][global_j] = grid.local_data[i + 1][j + 1];
            }
        }
        
        for (int src = 1; src < size; ++src) {
            int src_info[4];
            MPI_Recv(src_info, 4, MPI_INT, src, 0, comm, MPI_STATUS_IGNORE);
            
            int src_start_x = src_info[0];
            int src_start_y = src_info[1];
            int src_local_nx = src_info[2];
            int src_local_ny = src_info[3];
            
            std::vector<double> buffer(src_local_nx * src_local_ny);
            MPI_Recv(buffer.data(), buffer.size(), MPI_DOUBLE, src, 1, comm, MPI_STATUS_IGNORE);
            
            for (int i = 0; i < src_local_ny; ++i) {
                for (int j = 0; j < src_local_nx; ++j) {
                    int global_i = src_start_y + i;
                    int global_j = src_start_x + j;
                    global_grid[global_i][global_j] = buffer[i * src_local_nx + j];
                }
            }
        }
        
        std::ofstream file(filename);
        if (file.is_open()) {
            for (int i = 0; i < grid.global_ny; ++i) {
                for (int j = 0; j < grid.global_nx; ++j) {
                    file << std::fixed << std::setprecision(6) << global_grid[i][j];
                    if (j < grid.global_nx - 1) file << ",";
                }
                file << "\n";
            }
            file.close();
            std::cout << "Saved results to " << filename << std::endl;
        }
    } else {
        int info[4] = {grid.start_x, grid.start_y, grid.local_nx, grid.local_ny};
        MPI_Send(info, 4, MPI_INT, root, 0, comm);
        
        std::vector<double> buffer(grid.local_nx * grid.local_ny);
        for (int i = 0; i < grid.local_ny; ++i) {
            for (int j = 0; j < grid.local_nx; ++j) {
                buffer[i * grid.local_nx + j] = grid.local_data[i + 1][j + 1];
            }
        }
        MPI_Send(buffer.data(), buffer.size(), MPI_DOUBLE, root, 1, comm);
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Configuration
    const int global_nx = 100;
    const int global_ny = 100;
    const double Lx = 1.0;
    const double Ly = 1.0;
    const double alpha = 0.01;
    const int num_steps = 500;
    
    // Parse command line arguments for boundary conditions
    bool use_dirichlet = true; // Default to Dirichlet
    if (argc > 1) {
        std::string bc_type(argv[1]);
        if (bc_type == "neumann") {
            use_dirichlet = false;
        }
    }
    
    // Boundary conditions
    double left_bc = 100.0, right_bc = 0.0, top_bc = 50.0, bottom_bc = 0.0;
    
    int dims[2] = {0, 0};
    MPI_Dims_create(size, 2, dims);
    int periods[2] = {0, 0};
    int reorder = 1;
    
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cart_comm);
    
    int coords[2];
    MPI_Cart_coords(cart_comm, rank, 2, coords);
    
    if (rank == 0) {
        std::cout << "=== PARALLEL 2D HEAT SOLVER ===" << std::endl;
        std::cout << "Process grid: " << dims[0] << " x " << dims[1] << std::endl;
        std::cout << "Global grid: " << global_nx << " x " << global_ny << std::endl;
        std::cout << "Time steps: " << num_steps << std::endl;
        std::cout << "Boundary conditions: " << (use_dirichlet ? "Dirichlet" : "Neumann") << std::endl;
        if (use_dirichlet) {
            std::cout << "  Left: " << left_bc << "°C, Right: " << right_bc << "°C" << std::endl;
            std::cout << "  Top: " << top_bc << "°C, Bottom: " << bottom_bc << "°C" << std::endl;
        } else {
            std::cout << "  All boundaries: Insulated (zero flux)" << std::endl;
        }
    }
    
    DistributedGrid grid(global_nx, global_ny, dims[0], dims[1], coords[0], coords[1]);
    MPIHeatSolver2D solver(grid, Lx, Ly, alpha, left_bc, right_bc, top_bc, bottom_bc,
                          use_dirichlet, cart_comm);
    
    // Initialize with Gaussian hot spot
    solver.initialize_gaussian(0.5, 0.5, 200.0, 0.1);
    
    MPI_Barrier(cart_comm);
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Run simulation with non-blocking communication
    for (int step = 0; step < num_steps; ++step) {
        solver.step_nonblocking();
        
        if ((step + 1) % 100 == 0 && rank == 0) {
            std::cout << "Completed step " << (step + 1) << std::endl;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    if (rank == 0) {
        std::cout << "Simulation completed in " << duration.count() << " ms" << std::endl;
    }
    
    // Save results
    std::string filename = use_dirichlet ? "results/final_dirichlet.csv" : "results/final_neumann.csv";
    save_to_csv_parallel(grid, filename, cart_comm);
    
    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    
    return 0;
}

