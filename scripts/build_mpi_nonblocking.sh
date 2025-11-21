#!/bin/bash

# Create build directory
mkdir -p build
cd build

# Detect MPI compiler
if command -v mpic++ &> /dev/null; then
    MPI_COMPILER="mpic++"
elif command -v mpicxx &> /dev/null; then
    MPI_COMPILER="mpicxx"
else
    echo "Error: No MPI compiler found. Please install OpenMPI or MPICH."
    exit 1
fi

echo "Using MPI compiler: $MPI_COMPILER"

# Compile the serial version
echo "Compiling serial heat solver..."
g++ -std=c++11 -O2 -I../src ../src/heat_solver.cpp ../src/main.cpp -o heat_solver_serial

# Compile the MPI version
echo "Compiling MPI heat solver..."
$MPI_COMPILER -std=c++11 -O2 -I../src ../src/heat_solver_mpi_nonblocking.cpp ../src/main_mpi_nonblocking.cpp -o heat_solver_mpi_nonblocking

echo "Build complete!"
echo "Serial version: ./build/heat_solver_serial"
echo "MPI version:    mpirun -np 4 ./build/heat_solver_mpi_nonblocking"
