#!/usr/bin/env python3
"""
Test case generator for matrix computation problem:
E = A^T * B + C * D^T

Where:
- A: q×p matrix
- B: q×r matrix
- C: p×q matrix
- D: r×q matrix
- E: p×r matrix (output)

This script generates both near-maximum and highly rectangular test cases.
"""

import os
import random
import numpy as np

def generate_test_case(p, q, r, test_num, output_dir_input, output_dir_output):
    """
    Generate a single test case and compute the expected output.
    
    Args:
        p, q, r: Dimensions as per problem statement
        test_num: Test case number
        output_dir_input: Directory to save input files
        output_dir_output: Directory to save output files
    """
    
    # Generate matrices with random values in range [-10, 10]
    A = np.random.randint(-10, 11, size=(q, p))  # q×p
    B = np.random.randint(-10, 11, size=(q, r))  # q×r
    C = np.random.randint(-10, 11, size=(p, q))  # p×q
    D = np.random.randint(-10, 11, size=(r, q))  # r×q
    
    # Compute E = A^T * B + C * D^T
    E = np.dot(A.T, B) + np.dot(C, D.T)  # Result is p×r
    
    # Write input file
    input_filename = os.path.join(output_dir_input, f"input{test_num}.txt")
    with open(input_filename, 'w') as f:
        # First line: p, q, r
        f.write(f"{p} {q} {r}\n")
        
        # Next q lines: matrix A (q×p)
        for row in A:
            f.write(" ".join(map(str, row)) + "\n")
        
        # Next q lines: matrix B (q×r)
        for row in B:
            f.write(" ".join(map(str, row)) + "\n")
        
        # Next p lines: matrix C (p×q)
        for row in C:
            f.write(" ".join(map(str, row)) + "\n")
        
        # Next r lines: matrix D (r×q)
        for row in D:
            f.write(" ".join(map(str, row)) + "\n")
    
    # Write output file
    output_filename = os.path.join(output_dir_output, f"output{test_num}.txt")
    with open(output_filename, 'w') as f:
        # Write E as p×r matrix
        for row in E:
            f.write(" ".join(map(str, row)) + "\n")
    
    print(f"Generated test case {test_num}: p={p}, q={q}, r={r}")
    return A, B, C, D, E

def generate_near_max_cases(output_dir_input, output_dir_output):
    """Generate test cases with dimensions near maximum (1024)."""
    near_max_cases = [
        (1024, 1024, 1024),  # Maximum all dimensions
        (1000, 1024, 1000),  # Slightly below max in some dimensions
        (1024, 1000, 1024),
        (512, 1024, 512),    # Mixed: one dimension at max
        (1024, 512, 1024),
    ]
    
    for idx, (p, q, r) in enumerate(near_max_cases, 1):
        generate_test_case(p, q, r, idx, output_dir_input, output_dir_output)

def generate_rectangular_cases(output_dir_input, output_dir_output):
    """Generate highly rectangular (non-square) test cases."""
    rectangular_cases = [
        (2, 1024, 2),        # Very tall q dimension
        (1024, 2, 1024),     # Very small q dimension
        (10, 512, 100),      # Extreme aspect ratios
        (100, 512, 10),
        (2, 100, 512),       # Another extreme configuration
        (512, 100, 2),
    ]
    
    for idx, (p, q, r) in enumerate(rectangular_cases, 6):
        generate_test_case(p, q, r, idx, output_dir_input, output_dir_output)

def generate_small_test_cases(output_dir_input, output_dir_output):
    """Generate small test cases for verification."""
    small_cases = [
        (2, 2, 2),      # Minimal size
        (3, 4, 5),      # Small non-square
        (5, 3, 4),
    ]
    
    for idx, (p, q, r) in enumerate(small_cases, 12):
        generate_test_case(p, q, r, idx, output_dir_input, output_dir_output)

def verify_test_case(input_file, output_file):
    """
    Verify a test case by reading input and computing output.
    Compare with expected output.
    """
    # Read input file
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # Parse dimensions
    p, q, r = map(int, lines[0].split())
    
    # Parse matrices
    idx = 1
    A = []
    for i in range(q):
        A.append(list(map(int, lines[idx].split())))
        idx += 1
    A = np.array(A)
    
    B = []
    for i in range(q):
        B.append(list(map(int, lines[idx].split())))
        idx += 1
    B = np.array(B)
    
    C = []
    for i in range(p):
        C.append(list(map(int, lines[idx].split())))
        idx += 1
    C = np.array(C)
    
    D = []
    for i in range(r):
        D.append(list(map(int, lines[idx].split())))
        idx += 1
    D = np.array(D)
    
    # Compute E = A^T * B + C * D^T
    E_computed = np.dot(A.T, B) + np.dot(C, D.T)
    
    # Read expected output
    with open(output_file, 'r') as f:
        E_expected = []
        for line in f:
            E_expected.append(list(map(int, line.split())))
    E_expected = np.array(E_expected)
    
    # Verify
    if np.array_equal(E_computed, E_expected):
        print(f"✓ {os.path.basename(input_file)} verified successfully")
        return True
    else:
        print(f"✗ {os.path.basename(input_file)} verification FAILED")
        print(f"  Expected shape: {E_expected.shape}, Got: {E_computed.shape}")
        return False

def main():
    """Main function to generate all test cases."""
    
    # Create directories if they don't exist
    input_dir = "testcases/input"
    output_dir = "testcases/output"
    
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Matrix Computation Test Case Generator")
    print("Problem: E = A^T * B + C * D^T")
    print("=" * 60)
    
    print("\n[1/3] Generating near-maximum dimension test cases...")
    generate_near_max_cases(input_dir, output_dir)
    
    print("\n[2/3] Generating highly rectangular test cases...")
    generate_rectangular_cases(input_dir, output_dir)
    
    print("\n[3/3] Generating small test cases...")
    generate_small_test_cases(input_dir, output_dir)
    
    print("\n" + "=" * 60)
    print("Verifying generated test cases...")
    print("=" * 60)
    
    all_verified = True
    for i in range(1, 15):  # 14 total test cases
        input_file = os.path.join(input_dir, f"input{i}.txt")
        output_file = os.path.join(output_dir, f"output{i}.txt")
        
        if os.path.exists(input_file) and os.path.exists(output_file):
            if not verify_test_case(input_file, output_file):
                all_verified = False
    
    print("\n" + "=" * 60)
    if all_verified:
        print("✓ All test cases generated and verified successfully!")
    else:
        print("✗ Some test cases failed verification")
    print("=" * 60)

if __name__ == "__main__":
    main()
