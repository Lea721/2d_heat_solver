#!/bin/bash

# Create build directory
mkdir -p build
cd build

# Compile the code
echo "Compiling serial heat solver..."
g++ -std=c++11 -O2 -I../src ../src/heat_solver.cpp ../src/main.cpp -o heat_solver_serial

echo "Build complete!"
echo "Run with: ./build/heat_solver_serial"
