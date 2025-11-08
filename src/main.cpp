#include "heat_solver.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>

void save_to_csv(const std::vector<std::vector<double>>& data, 
                 const std::string& filename, int nx, int ny) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }
    
    // Write data in CSV format
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < nx; ++j) {
            file << std::fixed << std::setprecision(6) << data[i][j];
            if (j < nx - 1) file << ",";
        }
        file << "\n";
    }
    file.close();
    std::cout << "Saved results to " << filename << std::endl;
}

int main() {
    // Simulation parameters
    const int nx = 100;           // Grid points in x
    const int ny = 100;           // Grid points in y
    const double Lx = 1.0;        // Domain length in x
    const double Ly = 1.0;        // Domain length in y
    const double alpha = 0.01;    // Thermal diffusivity
    const int num_steps = 1000;   // Number of time steps
    const int save_interval = 100; // Save every N steps
    
    // Boundary conditions (Dirichlet)
    const double left_temp = 0.0;
    const double right_temp = 0.0;
    const double top_temp = 0.0;
    const double bottom_temp = 0.0;
    
    // Create solver
    HeatSolver2D solver(nx, ny, Lx, Ly, alpha, 
                       left_temp, right_temp, top_temp, bottom_temp,
                       true); // true for Dirichlet BCs
    
    // Initialize with a hot spot in the center
    solver.initialize_gaussian(0.5, 0.5, 1.0, 0.1);
    
    // Save initial condition
    save_to_csv(solver.get_temperature(), "results/initial.csv", nx, ny);
    
    // Run simulation
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < num_steps; ++step) {
        solver.step();
        
        if ((step + 1) % save_interval == 0) {
            std::string filename = "results/step_" + std::to_string(step + 1) + ".csv";
            save_to_csv(solver.get_temperature(), filename, nx, ny);
            std::cout << "Completed step " << (step + 1) << std::endl;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "Simulation completed in " << duration.count() << " ms" << std::endl;
    std::cout << "Final results saved to results/final.csv" << std::endl;
    
    // Save final state
    save_to_csv(solver.get_temperature(), "results/final.csv", nx, ny);
    
    return 0;
}
