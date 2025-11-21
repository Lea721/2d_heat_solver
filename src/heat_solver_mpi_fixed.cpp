#include "heat_solver_mpi.h"
#include <cmath>
#include <iostream>
#include <algorithm>

// ... (keep all the same code until exchange_halos_blocking)

void MPIHeatSolver2D::exchange_halos_blocking() {
    MPI_Status status;
    
    // FIX: Use MPI_Sendrecv with careful ordering to avoid deadlock
    // Exchange with east/west neighbors
    
    if (grid.east != MPI_PROC_NULL && grid.west != MPI_PROC_NULL) {
        // Both neighbors exist - use Sendrecv
        std::vector<double> send_east(grid.local_ny);
        std::vector<double> recv_west(grid.local_ny);
        std::vector<double> send_west(grid.local_ny);  
        std::vector<double> recv_east(grid.local_ny);
        
        // Prepare buffers
        for (int i = 0; i < grid.local_ny; ++i) {
            send_east[i] = grid.local_data[i + 1][grid.local_nx];  // Right boundary
            send_west[i] = grid.local_data[i + 1][1];              // Left boundary
        }
        
        // Exchange with east/west - careful ordering
        if (rank % 2 == 0) {
            // Even ranks send first, receive second
            MPI_Send(send_east.data(), grid.local_ny, MPI_DOUBLE, grid.east, 0, comm);
            MPI_Recv(recv_west.data(), grid.local_ny, MPI_DOUBLE, grid.west, 0, comm, &status);
            
            MPI_Send(send_west.data(), grid.local_ny, MPI_DOUBLE, grid.west, 1, comm);
            MPI_Recv(recv_east.data(), grid.local_ny, MPI_DOUBLE, grid.east, 1, comm, &status);
        } else {
            // Odd ranks receive first, send second  
            MPI_Recv(recv_west.data(), grid.local_ny, MPI_DOUBLE, grid.west, 0, comm, &status);
            MPI_Send(send_east.data(), grid.local_ny, MPI_DOUBLE, grid.east, 0, comm);
            
            MPI_Recv(recv_east.data(), grid.local_ny, MPI_DOUBLE, grid.east, 1, comm, &status);
            MPI_Send(send_west.data(), grid.local_ny, MPI_DOUBLE, grid.west, 1, comm);
        }
        
        // Unpack received data
        for (int i = 0; i < grid.local_ny; ++i) {
            grid.local_data[i + 1][0] = recv_west[i];                    // Left halo
            grid.local_data[i + 1][grid.local_nx + 1] = recv_east[i];    // Right halo
        }
    }
    
    // Exchange with north/south neighbors (similar pattern)
    if (grid.north != MPI_PROC_NULL && grid.south != MPI_PROC_NULL) {
        std::vector<double> send_north(grid.local_nx);
        std::vector<double> recv_south(grid.local_nx);
        std::vector<double> send_south(grid.local_nx);
        std::vector<double> recv_north(grid.local_nx);
        
        // Prepare buffers
        for (int j = 0; j < grid.local_nx; ++j) {
            send_north[j] = grid.local_data[grid.local_ny][j + 1];  // Top boundary
            send_south[j] = grid.local_data[1][j + 1];              // Bottom boundary
        }
        
        // Exchange with north/south
        if (rank % 2 == 0) {
            MPI_Send(send_north.data(), grid.local_nx, MPI_DOUBLE, grid.north, 2, comm);
            MPI_Recv(recv_south.data(), grid.local_nx, MPI_DOUBLE, grid.south, 2, comm, &status);
            
            MPI_Send(send_south.data(), grid.local_nx, MPI_DOUBLE, grid.south, 3, comm);
            MPI_Recv(recv_north.data(), grid.local_nx, MPI_DOUBLE, grid.north, 3, comm, &status);
        } else {
            MPI_Recv(recv_south.data(), grid.local_nx, MPI_DOUBLE, grid.south, 2, comm, &status);
            MPI_Send(send_north.data(), grid.local_nx, MPI_DOUBLE, grid.north, 2, comm);
            
            MPI_Recv(recv_north.data(), grid.local_nx, MPI_DOUBLE, grid.north, 3, comm, &status);
            MPI_Send(send_south.data(), grid.local_nx, MPI_DOUBLE, grid.south, 3, comm);
        }
        
        // Unpack received data
        for (int j = 0; j < grid.local_nx; ++j) {
            grid.local_data[0][j + 1] = recv_south[j];                    // Bottom halo
            grid.local_data[grid.local_ny + 1][j + 1] = recv_north[j];    // Top halo
        }
    }
    
    // Handle boundary processes (only one neighbor)
    // ... (similar pattern for processes on edges)
}

// ... (rest of the code remains the same)
