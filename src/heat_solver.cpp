#include "heat_solver.h"
#include <cmath>
#include <iostream>
#include <algorithm>

HeatSolver2D::HeatSolver2D(int nx, int ny, double Lx, double Ly, double alpha,
                         double left_temp, double right_temp,
                         double top_temp, double bottom_temp,
                         bool use_dirichlet)
    : nx(nx), ny(ny), alpha(alpha), use_dirichlet(use_dirichlet),
      left_bc(left_temp), right_bc(right_temp), 
      top_bc(top_temp), bottom_bc(bottom_temp) {
    
    // Calculate grid spacing
    dx = Lx / (nx - 1);
    dy = Ly / (ny - 1);
    
    // Initialize temperature fields
    T_old.resize(ny, std::vector<double>(nx, 0.0));
    T_new.resize(ny, std::vector<double>(nx, 0.0));
    
    // Calculate stable time step with safety factor
    dt = 0.25 * calculate_max_timestep();
    
    std::cout << "Grid: " << nx << " x " << ny << std::endl;
    std::cout << "Spacing: dx=" << dx << ", dy=" << dy << std::endl;
    std::cout << "Time step: " << dt << std::endl;
}

double HeatSolver2D::calculate_max_timestep() const {
    // Stability condition for explicit method
    return std::min(dx*dx, dy*dy) / (4.0 * alpha);
}

void HeatSolver2D::initialize_gaussian(double center_x, double center_y, 
                                     double amplitude, double spread) {
    for (int i = 0; i < ny; ++i) {
        double y = i * dy;
        for (int j = 0; j < nx; ++j) {
            double x = j * dx;
            double r2 = (x - center_x)*(x - center_x) + (y - center_y)*(y - center_y);
            T_old[i][j] = amplitude * exp(-r2 / (2.0 * spread * spread));
        }
    }
}

void HeatSolver2D::initialize_uniform(double temperature) {
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < nx; ++j) {
            T_old[i][j] = temperature;
        }
    }
}

void HeatSolver2D::step() {
    // Apply boundary conditions to old field
    if (use_dirichlet) {
        // Dirichlet BCs - fixed temperature
        for (int i = 0; i < ny; ++i) {
            T_old[i][0] = left_bc;        // Left boundary
            T_old[i][nx-1] = right_bc;    // Right boundary
        }
        for (int j = 0; j < nx; ++j) {
            T_old[0][j] = bottom_bc;      // Bottom boundary
            T_old[ny-1][j] = top_bc;      // Top boundary
        }
    } else {
        // Neumann BCs - zero flux (insulated)
        // Left boundary: T[-1][j] = T[1][j]
        for (int i = 0; i < ny; ++i) {
            T_old[i][0] = T_old[i][1];
        }
        // Right boundary: T[nx][j] = T[nx-2][j]
        for (int i = 0; i < ny; ++i) {
            T_old[i][nx-1] = T_old[i][nx-2];
        }
        // Bottom boundary: T[i][-1] = T[i][1]
        for (int j = 0; j < nx; ++j) {
            T_old[0][j] = T_old[1][j];
        }
        // Top boundary: T[ny][j] = T[ny-2][j]
        for (int j = 0; j < nx; ++j) {
            T_old[ny-1][j] = T_old[ny-2][j];
        }
    }
    
    // Update interior points using explicit finite difference
    for (int i = 1; i < ny - 1; ++i) {
        for (int j = 1; j < nx - 1; ++j) {
            double d2T_dx2 = (T_old[i][j-1] - 2.0 * T_old[i][j] + T_old[i][j+1]) / (dx * dx);
            double d2T_dy2 = (T_old[i-1][j] - 2.0 * T_old[i][j] + T_old[i+1][j]) / (dy * dy);
            
            T_new[i][j] = T_old[i][j] + alpha * dt * (d2T_dx2 + d2T_dy2);
        }
    }
    
    swap_fields();
}

void HeatSolver2D::swap_fields() {
    std::swap(T_old, T_new);
}
