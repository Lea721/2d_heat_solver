#include "heat_solver_mpi_nonblocking.h"
#include "visualizer_opengl.h"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <mpi.h>
#include <vector>
#include <unistd.h>   // for usleep()

// ==========================================================
// CORRECTED GATHER FUNCTION — FIXES TILED VISUALIZATION
// ==========================================================

void gather_global_grid(const DistributedGrid& grid,
                        std::vector<std::vector<double>>& global_grid,
                        MPI_Comm comm, int root = 0)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (rank == root) {
        // Allocate full grid
        global_grid.assign(
            grid.global_ny,
            std::vector<double>(grid.global_nx, 0.0)
        );

        // Copy OWN block
        for (int i = 0; i < grid.local_ny; ++i)
            for (int j = 0; j < grid.local_nx; ++j)
                global_grid[grid.start_y + i][grid.start_x + j] =
                    grid.local_data[i + 1][j + 1];

        // Receive other blocks
        for (int src = 1; src < size; ++src)
        {
            int info[4];
            MPI_Recv(info, 4, MPI_INT, src, 0, comm, MPI_STATUS_IGNORE);

            int sx = info[0];   // start_x
            int sy = info[1];   // start_y
            int nx = info[2];   // local_nx
            int ny = info[3];   // local_ny

            std::vector<double> buffer(nx * ny);
            MPI_Recv(buffer.data(), nx * ny, MPI_DOUBLE, src, 1, comm,
                     MPI_STATUS_IGNORE);

            // CORRECT INDEXING — FIXES REPEATING TILES
            for (int i = 0; i < ny; i++)
                for (int j = 0; j < nx; j++)
                    global_grid[sy + i][sx + j] =
                        buffer[i * nx + j];
        }
    }

    else {
        // Send metadata: start_x, start_y, local_nx, local_ny
        int info[4] = {
            grid.start_x, grid.start_y,
            grid.local_nx, grid.local_ny
        };
        MPI_Send(info, 4, MPI_INT, root, 0, comm);

        // Send actual data
        std::vector<double> buffer(grid.local_nx * grid.local_ny);
        for (int i = 0; i < grid.local_ny; i++)
            for (int j = 0; j < grid.local_nx; j++)
                buffer[i * grid.local_nx + j] =
                    grid.local_data[i + 1][j + 1];

        MPI_Send(buffer.data(), buffer.size(), MPI_DOUBLE, root, 1, comm);
    }
}


// ==========================================================
// MAIN PROGRAM (WITH REAL-TIME OPENGL VISUALIZATION)
// ==========================================================

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // --------------------
    // Simulation parameters
    // --------------------
    const int global_nx = 300;
    const int global_ny = 300;

    const double Lx = 1.0, Ly = 1.0;
    const double alpha = 0.01;

    const int num_steps = 500;

    const double left_temp = 100.0;
    const double right_temp = 0.0;
    const double top_temp = 50.0;
    const double bottom_temp = 0.0;

    // --------------------
    // Create MPI Cartesian grid
    // --------------------
    int dims[2] = {0, 0};
    MPI_Dims_create(size, 2, dims);

    int periods[2] = {0, 0};  // no periodic boundaries

    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart_comm);

    int coords[2];
    MPI_Cart_coords(cart_comm, rank, 2, coords);

    if (rank == 0) {
        std::cout << "\n=== MPI HEAT SOLVER + REAL-TIME OPENGL VISUALIZATION ===\n";
        std::cout << "Process grid: " << dims[0] << " x " << dims[1] << "\n";
        std::cout << "Grid: " << global_nx << "x" << global_ny << "\n";
        std::cout << "Time steps: " << num_steps << "\n\n";
    }

    // --------------------
    // Initialize solver
    // --------------------
    DistributedGrid grid(global_nx, global_ny,
                         dims[0], dims[1],
                         coords[0], coords[1]);

    MPIHeatSolver2D solver(grid, Lx, Ly, alpha,
                           left_temp, right_temp, top_temp, bottom_temp,
                           true, cart_comm);

    solver.initialize_gaussian(0.5, 0.5, 200.0, 0.1);

    // --------------------
    // OpenGL init on rank 0
    // --------------------
    if (rank == 0) {
        visualizer::init(argc, argv, global_nx, global_ny, 800, 800);
    }

    MPI_Barrier(cart_comm);
    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<std::vector<double>> global_grid;


    // ================================================================
    // MAIN SIMULATION LOOP (each timestep = 1 frame)
    // ================================================================
    for (int step = 0; step < num_steps; step++) {

        solver.step_nonblocking();

        // Gather full grid into rank 0
        gather_global_grid(grid, global_grid, cart_comm, 0);

        if (rank == 0) {

            // Flatten 2D → 1D array for OpenGL
            std::vector<double> flat(global_nx * global_ny);

            for (int i = 0; i < global_ny; i++)
                for (int j = 0; j < global_nx; j++)
                    flat[i * global_nx + j] = global_grid[i][j];

            // Push data to OpenGL
            visualizer::update_grid(flat);
            visualizer::poll_and_draw();

            // Control framerate (60 fps)
            usleep(16000);

            if ((step + 1) % 50 == 0)
                std::cout << "Step " << (step + 1)
                          << " / " << num_steps << "\n";
        }
    }

    if (rank == 0)
        visualizer::finish();

    MPI_Finalize();
    return 0;
}

