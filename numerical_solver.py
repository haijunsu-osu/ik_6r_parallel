#!/usr/bin/env python3
"""
Numerical Solver for Trigonometric System

Solves the system: A[cos θ₁, sin θ₁] + B[cos θ₂, sin θ₂] = C

This script implements the complete algorithm:
1. Express cos(θ₂) and sin(θ₂) in terms of cos(θ₁) and sin(θ₁) 
2. Apply the identity cos²(θ₂) + sin²(θ₂) = 1
3. Use Weierstrass substitution to get a quartic polynomial in t
4. Solve for t using numpy.roots
5. Convert back to θ₁ and θ₂

Extended functionality for singular B matrices:
- Handles rank-deficient matrices 
- Direct solving for zero matrix case
- Geometric constraints for rank-1 matrices

Dependencies: numpy, math only

Author: Haijun Su with Assistance from GitHub Copilot 
Date: September 27, 2025
"""

import numpy as np
import math

import numpy as np
from utilities import solve_trig_eq, solve_one_equation_for_two_unknowns, solve_trig_sys_single, pre_process


def solve_zero_b(A: np.ndarray, C: np.ndarray, verbose: bool = False) -> list:
    """
    Solve the case where B = 0, so the system becomes: A[cos(θ₁), sin(θ₁)] = C
    
    This is a 2x2 linear system in cos(θ₁) and sin(θ₁).
    """
    solutions = []
    
    if verbose:
        print("\nCase: B = 0 matrix")
        print("System reduces to: A[cos(θ₁), sin(θ₁)] = C")
    
    # Use the new single variable solver
    th1_solutions = solve_trig_sys_single(A, C, verbose)
    
    # Convert to the expected format with θ₂ as free parameter
    for sol in th1_solutions:
        # For B=0, θ₂ is free, set it to 0 as default
        th2 = 0.0
        solutions.append({
            'th1': sol['th'],
            'th2': th2,
            'note': 'θ₂ is free (zero B matrix)'
        })
    
    return solutions


def solve_rank_1_b(A: np.ndarray, B: np.ndarray, C: np.ndarray, 
                   verbose: bool = False) -> list:
    """
    Solve the case where B has rank 1.
    
    After preprocessing (row reduction), the second row of B becomes [0, 0],
    so the second equation only involves θ₁:
    A[1,0]*cos(θ₁) + A[1,1]*sin(θ₁) = C[1]
    
    Strategy:
    1. Solve the second equation for θ₁ using solve_trig_eq()
    2. For each θ₁ solution, substitute into the first equation to solve for θ₂
    """
    solutions = []
    
    if verbose:
        print("\nCase: B has rank 1 - using row reduction approach")
        print("After preprocessing, second row of B is [0, 0]")
        print("Second equation: A[1,0]*cos(θ₁) + A[1,1]*sin(θ₁) = C[1]")
    
    # The second equation only involves θ₁: A[1,0]*cos(θ₁) + A[1,1]*sin(θ₁) = C[1]
    # Rearrange as: A[1,0]*cos(θ₁) + A[1,1]*sin(θ₁) - C[1] = 0
    a_coeff = A[1, 0]
    b_coeff = A[1, 1] 
    c_coeff = -C[1]
    
    # Check if the second equation is 0=0 (trivial equation)
    coeff_magnitude = math.sqrt(a_coeff**2 + b_coeff**2)
    if coeff_magnitude < 1e-12 and abs(C[1]) < 1e-12:
        if verbose:
            print("Second equation is 0=0 (trivial) - solving first equation with two unknowns")
        
        # The second equation gives no constraint, so we solve the first equation
        # which has the form: A[0,0]*cos(θ₁) + A[0,1]*sin(θ₁) + B[0,0]*cos(θ₂) + B[0,1]*sin(θ₂) = C[0]

        return solve_one_equation_for_two_unknowns(A[0, 0], A[0, 1], B[0, 0], B[0, 1], C[0])

    if verbose:
        print(f"Equation: {a_coeff:.6f}*cos(θ₁) + {b_coeff:.6f}*sin(θ₁) + {c_coeff:.6f} = 0")
    
    # Solve for θ₁ using solve_trig_eq
    th1_solutions, th1_arbitrary, _ = solve_trig_eq(a_coeff, b_coeff, c_coeff, verbose)
    
    if verbose:
        print(f"Found {len(th1_solutions)} solutions for θ₁")
        if th1_arbitrary:
            print("θ₁ is arbitrary (0=0 equation)")
    
    # For each θ₁ solution, solve for θ₂ using the first equation
    for i, th1 in enumerate(th1_solutions):
        cos_th1 = math.cos(th1)
        sin_th1 = math.sin(th1)
        
        if verbose:
            print(f"\nSolving for θ₂ with θ₁ = {th1:.6f}")
            print(f"cos(θ₁) = {cos_th1:.6f}, sin(θ₁) = {sin_th1:.6f}")
        
        # Substitute θ₁ into the first equation:
        # A[0,0]*cos(θ₁) + A[0,1]*sin(θ₁) + B[0,0]*cos(θ₂) + B[0,1]*sin(θ₂) = C[0]
        # Rearrange: B[0,0]*cos(θ₂) + B[0,1]*sin(θ₂) = C[0] - A[0,0]*cos(θ₁) - A[0,1]*sin(θ₁)
        
        rhs = C[0] - A[0, 0] * cos_th1 - A[0, 1] * sin_th1

        if verbose:
            print(f"First equation becomes: {B[0,0]:.6f}*cos(θ₂) + {B[0,1]:.6f}*sin(θ₂) = {rhs:.6f}")
            print(f"Rearranged: {B[0,0]:.6f}*cos(θ₂) + {B[0,1]:.6f}*sin(θ₂) + {-rhs:.6f} = 0")

        # Solve for θ₂ using solve_trig_eq
        th2_solutions, th2_arbitrary, _ = solve_trig_eq(B[0, 0], B[0, 1], -rhs, verbose)

        if verbose:
            print(f"Found {len(th2_solutions)} solutions for θ₂")
            if th2_arbitrary:
                print("θ₂ is arbitrary (0=0 equation)")

        # Add valid solutions
        for th2 in th2_solutions:
            cos_th2 = math.cos(th2)
            sin_th2 = math.sin(th2)

            # Verify the solution satisfies the original system
            residual = A @ np.array([cos_th1, sin_th1]) + B @ np.array([cos_th2, sin_th2]) - C
            if np.linalg.norm(residual) < 1e-8:
                solutions.append({
                    'th1': th1,
                    'th2': th2
                })
                if verbose:
                    print(f"  ✓ Valid solution: θ₁ = {th1:.6f}, θ₂ = {th2:.6f}")
            elif verbose:
                print(f"  ✗ Invalid solution (residual = {np.linalg.norm(residual):.2e})")
    
    if verbose:
        print(f"\nTotal solutions found: {len(solutions)}")
    
    return solutions

def solve_quartic(A, B, C, verbose=False, real_solutions_only=True):
    """
    Solve the trigonometric system using the quartic polynomial method.
    
    This function handles the case where B is non-singular, using the standard
    quartic polynomial approach with Weierstrass substitution.
    
    Args:
        A: 2x2 numpy array
        B: 2x2 numpy array (non-singular)
        C: 2x1 numpy array
        verbose: Print detailed steps
        real_solutions_only: If True, only return real solutions
        
    Returns:
        List of solution dictionaries
    """
    solutions = []
    
    if verbose:
        print("Matrix B is non-singular, using standard quartic polynomial method")
    
    # Extract matrix elements for clarity
    A00, A01 = A[0, 0], A[0, 1]
    A10, A11 = A[1, 0], A[1, 1]
    B00, B01 = B[0, 0], B[0, 1]
    B10, B11 = B[1, 0], B[1, 1]
    C0, C1 = C[0], C[1]
    
    if verbose:
        print(f"\nStep 1: Express cos(θ₂) and sin(θ₂) in terms of cos(θ₁) and sin(θ₁)")
        print(f"From B[cos(θ₂), sin(θ₂)]^T = C - A[cos(θ₁), sin(θ₁)]^T")
    
    # Step 2: Build the quartic polynomial coefficients directly
    # This implements the full symbolic derivation numerically
    
    if verbose:
        print(f"\nStep 2: Building quartic polynomial coefficients")
    
    # These coefficients come from the symbolic derivation:
    # After applying Weierstrass substitution and trigonometric identity
    
    # Denominator for all coefficients  
    denom = np.linalg.det(B)**2
    
    # Common terms that appear in the coefficients
    A00_sq = A00**2
    A01_sq = A01**2  
    A10_sq = A10**2
    A11_sq = A11**2
    B00_sq = B00**2
    B01_sq = B01**2
    B10_sq = B10**2
    B11_sq = B11**2
    C0_sq = C0**2
    C1_sq = C1**2
    
    # Build the numerators for each coefficient
    # a4 coefficient (t^4 term)
    a4_num = (A00_sq*B10_sq + A00_sq*B11_sq - 2*A00*A10*B00*B10 - 2*A00*A10*B01*B11 
              - 2*A00*B00*B10*C1 - 2*A00*B01*B11*C1 + 2*A00*B10_sq*C0 + 2*A00*B11_sq*C0 
              + A10_sq*B00_sq + A10_sq*B01_sq + 2*A10*B00_sq*C1 - 2*A10*B00*B10*C0 
              + 2*A10*B01_sq*C1 - 2*A10*B01*B11*C0 - B00_sq*B11_sq + B00_sq*C1_sq 
              + 2*B00*B01*B10*B11 - 2*B00*B10*C0*C1 - B01_sq*B10_sq + B01_sq*C1_sq 
              - 2*B01*B11*C0*C1 + B10_sq*C0_sq + B11_sq*C0_sq)
    
    # a3 coefficient (t^3 term)
    a3_num = 4*(-A00*A01*B10_sq - A00*A01*B11_sq + A00*A11*B00*B10 + A00*A11*B01*B11 
                + A01*A10*B00*B10 + A01*A10*B01*B11 + A01*B00*B10*C1 + A01*B01*B11*C1 
                - A01*B10_sq*C0 - A01*B11_sq*C0 - A10*A11*B00_sq - A10*A11*B01_sq 
                - A11*B00_sq*C1 + A11*B00*B10*C0 - A11*B01_sq*C1 + A11*B01*B11*C0)
    
    # a2 coefficient (t^2 term)  
    a2_num = 2*(-A00_sq*B10_sq - A00_sq*B11_sq + 2*A00*A10*B00*B10 + 2*A00*A10*B01*B11 
                + 2*A01_sq*B10_sq + 2*A01_sq*B11_sq - 4*A01*A11*B00*B10 - 4*A01*A11*B01*B11 
                - A10_sq*B00_sq - A10_sq*B01_sq + 2*A11_sq*B00_sq + 2*A11_sq*B01_sq 
                - B00_sq*B11_sq + B00_sq*C1_sq + 2*B00*B01*B10*B11 - 2*B00*B10*C0*C1 
                - B01_sq*B10_sq + B01_sq*C1_sq - 2*B01*B11*C0*C1 + B10_sq*C0_sq + B11_sq*C0_sq)
    
    # a1 coefficient (t^1 term)
    a1_num = 4*(A00*A01*B10_sq + A00*A01*B11_sq - A00*A11*B00*B10 - A00*A11*B01*B11 
                - A01*A10*B00*B10 - A01*A10*B01*B11 + A01*B00*B10*C1 + A01*B01*B11*C1 
                - A01*B10_sq*C0 - A01*B11_sq*C0 + A10*A11*B00_sq + A10*A11*B01_sq 
                - A11*B00_sq*C1 + A11*B00*B10*C0 - A11*B01_sq*C1 + A11*B01*B11*C0)
    
    # a0 coefficient (constant term)
    a0_num = (A00_sq*B10_sq + A00_sq*B11_sq - 2*A00*A10*B00*B10 - 2*A00*A10*B01*B11 
              + 2*A00*B00*B10*C1 + 2*A00*B01*B11*C1 - 2*A00*B10_sq*C0 - 2*A00*B11_sq*C0 
              + A10_sq*B00_sq + A10_sq*B01_sq - 2*A10*B00_sq*C1 + 2*A10*B00*B10*C0 
              - 2*A10*B01_sq*C1 + 2*A10*B01*B11*C0 - B00_sq*B11_sq + B00_sq*C1_sq 
              + 2*B00*B01*B10*B11 - 2*B00*B10*C0*C1 - B01_sq*B10_sq + B01_sq*C1_sq 
              - 2*B01*B11*C0*C1 + B10_sq*C0_sq + B11_sq*C0_sq)
    
    # Final coefficients
    a4 = a4_num / denom
    a3 = a3_num / denom  
    a2 = a2_num / denom
    a1 = a1_num / denom
    a0 = a0_num / denom
    
    if verbose:
        print(f"Quartic polynomial: {a4:.6f}*t^4 + {a3:.6f}*t^3 + {a2:.6f}*t^2 + {a1:.6f}*t + {a0:.6f} = 0")
    
    # Form coefficient array for numpy.roots (highest degree first)
    coeffs = [a4, a3, a2, a1, a0]
    
    # Remove leading zeros for numerical stability
    while len(coeffs) > 1 and abs(coeffs[0]) < 1e-12:
        coeffs = coeffs[1:]
    
    if len(coeffs) == 1:
        if abs(coeffs[0]) < 1e-12:
            if verbose:
                print("Warning: All coefficients are zero - infinite solutions")
            return solutions
        else:
            if verbose:
                print("Warning: No solutions - constant polynomial")
            return solutions
    
    # Step 3: Solve quartic polynomial for t
    if verbose:
        print(f"\nStep 3: Solving polynomial of degree {len(coeffs)-1}")
    
    try:
        t_roots = np.roots(coeffs)
    except np.linalg.LinAlgError:
        if verbose:
            print("Warning: Failed to find polynomial roots")
        return solutions
    
    if verbose:
        print(f"Found {len(t_roots)} roots: {t_roots}")
    
    # Step 4: Process each root
    for i, t in enumerate(t_roots):
        if verbose:
            print(f"\nProcessing root {i+1}: t = {t}")
            
        # Skip complex roots with significant imaginary part (if real_solutions_only is True)
        if np.iscomplex(t) and abs(t.imag) > 1e-10:
            if real_solutions_only:
                if verbose:
                    print("  Skipping complex root (real_solutions_only=True)")
                continue
            else:
                if verbose:
                    print("  Processing complex root (real_solutions_only=False)")
                # For complex roots, we'll work with complex arithmetic throughout
            
        # Convert t to real if it's effectively real, but keep complex if needed
        if np.iscomplex(t) and abs(t.imag) <= 1e-10:
            t = t.real
        
        # Check for extreme t values that could cause numerical issues
        if abs(t) > 1e8:
            if verbose:
                print(f"  Skipping extreme t value: {t}")
            continue
        
        # Step 5: Convert t back to cos(θ₁) and sin(θ₁) using inverse Weierstrass substitution
        denominator = 1 + t**2
        cos_th1 = (1 - t**2) / denominator
        sin_th1 = 2*t / denominator
        
        if verbose:
            if np.iscomplex(cos_th1) or np.iscomplex(sin_th1):
                print(f"  cos(θ₁) = {cos_th1}, sin(θ₁) = {sin_th1}")
            else:
                print(f"  cos(θ₁) = {cos_th1:.6f}, sin(θ₁) = {sin_th1:.6f}")
        
        # Verify trigonometric identity (allow complex values if real_solutions_only=False)
        identity_check = cos_th1**2 + sin_th1**2 - 1
        if abs(identity_check) > 1e-4:
            if verbose:
                print(f"  Failed trigonometric identity check for θ₁ (error: {identity_check})")
            continue
            
        # Calculate θ₁ from cos_th1 and sin_th1 (handle complex case)
        if np.iscomplex(cos_th1) or np.iscomplex(sin_th1):
            if real_solutions_only:
                if verbose:
                    print("  Skipping complex trigonometric values (real_solutions_only=True)")
                continue
            else:
                # For complex case, use complex atan2 equivalent
                th1 = np.log((cos_th1 + 1j * sin_th1) / np.sqrt(cos_th1**2 + sin_th1**2)) / 1j
                if np.iscomplex(th1):
                    th1 = th1.real if abs(th1.imag) < 1e-10 else th1
        else:
            th1 = math.atan2(float(sin_th1.real), float(cos_th1.real))
        
        # Step 6: Solve for cos(θ₂) and sin(θ₂) using the original equations
        # B * [cos_th2, sin_th2]^T = C - A * [cos_th1, sin_th1]^T
        rhs = C - A @ np.array([cos_th1, sin_th1])
        
        try:
            trig_th2 = np.linalg.solve(B, rhs)
            cos_th2, sin_th2 = trig_th2[0], trig_th2[1]
        except np.linalg.LinAlgError:
            if verbose:
                print("  Failed to solve for cos(θ₂), sin(θ₂)")
            continue
            
        if verbose:
            if np.iscomplex(cos_th2) or np.iscomplex(sin_th2):
                print(f"  cos(θ₂) = {cos_th2}, sin(θ₂) = {sin_th2}")
            else:
                print(f"  cos(θ₂) = {cos_th2:.6f}, sin(θ₂) = {sin_th2:.6f}")
            
        # Verify trigonometric identity for θ₂ (allow complex values if real_solutions_only=False)
        identity_check_th2 = cos_th2**2 + sin_th2**2 - 1
        if abs(identity_check_th2) > 1e-4:
            if verbose:
                print(f"  Failed trigonometric identity check for θ₂ (error: {identity_check_th2})")
            continue
            
        # Calculate θ₂ from cos_th2 and sin_th2 (handle complex case)
        if np.iscomplex(cos_th2) or np.iscomplex(sin_th2):
            if real_solutions_only:
                if verbose:
                    print("  Skipping complex trigonometric values for θ₂ (real_solutions_only=True)")
                continue
            else:
                # For complex case, use complex atan2 equivalent
                th2 = np.log((cos_th2 + 1j * sin_th2) / np.sqrt(cos_th2**2 + sin_th2**2)) / 1j
                if np.iscomplex(th2):
                    th2 = th2.real if abs(th2.imag) < 1e-10 else th2
        else:
            th2 = math.atan2(float(sin_th2.real), float(cos_th2.real))
        
        # Step 7: Verify the original equations
        # This checks if the solution satisfies A[cos θ₁, sin θ₁] + B[cos θ₂, sin θ₂] = C
        # Multiple valid solutions are expected for trigonometric systems
        eq1_residual = A[0,0]*cos_th1 + A[0,1]*sin_th1 + B[0,0]*cos_th2 + B[0,1]*sin_th2 - C[0]
        eq2_residual = A[1,0]*cos_th1 + A[1,1]*sin_th1 + B[1,0]*cos_th2 + B[1,1]*sin_th2 - C[1]
        
        if verbose:
            if np.iscomplex(eq1_residual) or np.iscomplex(eq2_residual):
                print(f"  Equation residuals: {eq1_residual}, {eq2_residual}")
            else:
                print(f"  Equation residuals: {eq1_residual:.2e}, {eq2_residual:.2e}")
        
        if abs(eq1_residual) < 1e-10 and abs(eq2_residual) < 1e-10:
            solution_dict = {
                'th1': th1,
                'th2': th2,
                'cos_th1': cos_th1,
                'sin_th1': sin_th1,
                'cos_th2': cos_th2,
                'sin_th2': sin_th2,
                't': t
            }
            
            # Add flag to indicate if solution is complex
            is_complex_solution = any(np.iscomplex(val) for val in [th1, th2, cos_th1, sin_th1, cos_th2, sin_th2])
            solution_dict['is_complex'] = is_complex_solution
            
            solutions.append(solution_dict)
            
            if verbose:
                solution_type = "complex" if is_complex_solution else "real"
                if np.iscomplex(th1) or np.iscomplex(th2):
                    print(f"  ✓ Valid {solution_type} solution: θ₁ = {th1}, θ₂ = {th2}")
                else:
                    print(f"  ✓ Valid {solution_type} solution: θ₁ = {th1:.6f}, θ₂ = {th2:.6f}")
        elif verbose:
            print("  ✗ Failed equation verification")
    
    return solutions


def solve_trig_sys(A, B, C, verbose=False, real_solutions_only=True):
    """
    Solve the trigonometric system A[cos θ₁, sin θ₁] + B[cos θ₂, sin θ₂] = C.
    
    This function automatically handles both regular and singular B matrices.
    
    Args:
        A: 2x2 numpy array
        B: 2x2 numpy array (may be singular)
        C: 2x1 numpy array
        verbose: Print detailed steps (default: False for cleaner output)
        real_solutions_only: If True (default), only return real solutions. 
                           If False, also include complex solutions.
        
    Returns:
        List of solution dictionaries, each containing 'th1', 'th2', 'cos_th1', 'sin_th1', 'cos_th2', 'sin_th2'
    """
    solutions = []
    
    # Input validation
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float) 
    C = np.asarray(C, dtype=float)
    
    if A.shape != (2, 2) or B.shape != (2, 2) or C.shape != (2,):
        raise ValueError("Invalid input dimensions: A and B must be 2x2, C must be 2x1")
       
    if verbose:
        print("Solving system: A[cos θ₁, sin θ₁] + B[cos θ₂, sin θ₂] = C")
        print(f"A = \n{A}")
        print(f"B = \n{B}") 
        print(f"C = {C}")
    
    # Preprocess the matrices
    rank_A, rank_B, A_norm, B_norm, C_norm = pre_process(A, B, C, verbose)
    
    # Use the normalized/preprocessed matrices for solving
    A, B, C = A_norm, B_norm, C_norm
    
    # Dispatch based on matrix ranks
    if rank_A == 0:
        # A is zero matrix - θ₁ is free, solve B*[cos(θ₂), sin(θ₂)] = C
        if verbose:
            print("A is zero matrix - θ₁ is free, solving for θ₂ only")
        
        # Use solve_trig_sys_single to solve B*[cos(θ₂), sin(θ₂)] = C
        th2_solutions = solve_trig_sys_single(B, C, verbose)
        
        # For each θ₂ solution, θ₁ is free (set to 0 as default)
        solutions = []
        for sol in th2_solutions:
            solutions.append({
                'th1': 0.0,  # θ₁ is free
                'th2': sol['th'],
                'cos_th1': 1.0,
                'sin_th1': 0.0,
                'cos_th2': sol['cos_th'],
                'sin_th2': sol['sin_th'],
                'note': 'θ₁ is free (zero A matrix)'
            })
        
    elif rank_B == 0:
        # B is zero matrix - solve for θ₁ only
        solutions = solve_zero_b(A, C, verbose)
    elif rank_B == 1:
        # B has rank 1 - use null space approach
        solutions = solve_rank_1_b(A, B, C, verbose)
    elif rank_B == 2:
        # B is full rank - use quartic polynomial method
        solutions = solve_quartic(A, B, C, verbose, real_solutions_only)
    else:
        # Unexpected rank
        if verbose:
            print(f"Unexpected rank_B = {rank_B}")
        return []
    
    # Validate all solutions by substituting back into the original equations
    validated_solution_arrays = []  # List[np.ndarray([th1, th2])]
    validated_flags = []  # List[List[int]] parallel list indicating free angles [f1, f2]

    for i, sol in enumerate(solutions):
        # Extract solution values (fall back to computed cos/sin if angles not provided)
        th1 = sol['th1']
        th2 = sol['th2']
        cos_th1 = sol.get('cos_th1', math.cos(th1))
        sin_th1 = sol.get('sin_th1', math.sin(th1))
        cos_th2 = sol.get('cos_th2', math.cos(th2))
        sin_th2 = sol.get('sin_th2', math.sin(th2))

        # Compute the residual: A[cos(θ₁), sin(θ₁)] + B[cos(θ₂), sin(θ₂)] - C
        residual = A @ np.array([cos_th1, sin_th1]) + B @ np.array([cos_th2, sin_th2]) - C
        residual_norm = np.linalg.norm(residual)

        # Check if solution satisfies the equations within tolerance
        tolerance = 1e-5
        if residual_norm < tolerance:
            # Pack solution as numpy array
            sol_arr = np.array([float(th1), float(th2)])

            # Determine free-angle flags based on 'note' or other indicators (default: not free)
            flags = [0, 0]
            note = sol.get('note', '')
            if 'θ₁' in note and 'free' in note:
                flags[0] = 1
            if 'θ₂' in note and 'free' in note:
                flags[1] = 1

            validated_solution_arrays.append(sol_arr)
            validated_flags.append(flags)

            if verbose:
                print(f"Solution {i+1}: ✓ Valid (residual = {residual_norm:.2e})")
        else:
            if verbose:
                print(f"Solution {i+1}: ✗ Invalid (residual = {residual_norm:.2e})")
                print(f"  Residual vector: {residual}")

    if verbose:
        print(f"\nValidation complete: {len(validated_solution_arrays)}/{len(solutions)} solutions passed")

    return validated_solution_arrays, validated_flags
