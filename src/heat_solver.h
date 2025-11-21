#ifndef HEAT_SOLVER_H
#define HEAT_SOLVER_H

#include <vector>

class HeatSolver2D {
private:
    // Grid parameters
    int nx, ny;           // Grid dimensions
    double dx, dy;        // Grid spacing
    double dt;            // Time step
    double alpha;         // Thermal diffusivity
    
    // Temperature fields
    std::vector<std::vector<double>> T_old;
    std::vector<std::vector<double>> T_new;
    
    // Boundary conditions
    double left_bc, right_bc, top_bc, bottom_bc;
    bool use_dirichlet;   // true for Dirichlet, false for Neumann
    
public:
    // Constructor
    HeatSolver2D(int nx, int ny, double Lx, double Ly, double alpha, 
                 double left_temp, double right_temp, 
                 double top_temp, double bottom_temp,
                 bool use_dirichlet = true);
    
    // Initialize temperature field
    void initialize_gaussian(double center_x, double center_y, 
                         double amplitude = 1.0, double spread = 0.1);
    void initialize_uniform(double temperature);
    
    // Simulation step
    void step();
    void copy_old_to_new();
    // Getters
    const std::vector<std::vector<double>>& get_temperature() const { return T_new; }
    int get_nx() const { return nx; }
    int get_ny() const { return ny; }
    
    // Utility functions
    void swap_fields();
    double calculate_max_timestep() const;
};

#endif
