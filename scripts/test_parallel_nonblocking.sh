#!/bin/bash

# Test script for parallel heat solver

echo "Testing Parallel 2D Heat Solver"
echo "================================"

# Create results directory
mkdir -p results

# Run with different numbers of processes
for np in 1 2 4; do
    echo ""
    echo "Running with $np processes..."
    mpirun -np $np ./build/heat_solver_mpi_nonblocking
    
    if [ $? -eq 0 ]; then
        echo "✓ Success with $np processes"
    else
        echo "✗ Failed with $np processes"
        exit 1
    fi
done

echo ""
echo "All tests completed successfully!"
echo "Check results/ directory for output files"
