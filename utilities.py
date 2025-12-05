"""
DH Matrix, Screw Motion and Analytical Solvers for Algebraic Systems

This module provides functions for computing DH transformation matrices and screw motions
for robotic kinematics, as well as analytical solvers for algebraic systems.

Author: Haijun Su with Assistance from GitHub Copilot
Date: September 27, 2025
"""

import numpy as np
import math
from scipy.spatial.transform import Rotation
from typing import List, Tuple

"""Optional import of robust two-angle solver.

We avoid using a package-qualified name (scripts.two_angle_solver) because the
"scripts" directory may not contain an __init__.py. Instead we append the
directory to sys.path and import the module directly. All symbols are set to
None when import fails so downstream logic can fall back gracefully.
"""
import os as _os, sys as _sys, importlib as _importlib
_EqCoeffs = None
_TwoEqSystem = None
_solve_two_angle_system = None
_SolverDiag = None
_eval_two_eqs = None
try:
    _repo_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
    _scripts_dir = _os.path.join(_repo_root, "scripts")
    if _scripts_dir not in _sys.path:
        _sys.path.append(_scripts_dir)
    _mod = _importlib.import_module('two_angle_solver')
    _EqCoeffs = getattr(_mod, 'EquationCoefficients', None)
    _TwoEqSystem = getattr(_mod, 'TwoEquationSystem', None)
    _solve_two_angle_system = getattr(_mod, 'solve_two_angle_system', None)
    _SolverDiag = getattr(_mod, 'SolverDiagnostics', None)
    _eval_two_eqs = getattr(_mod, 'evaluate_equations', None)
except Exception:
    pass


def ScrewX(a, alpha):
    """
    Screw motion along X-axis: Translation by 'a' along X, then rotation by 'alpha' about X.

    ScrewX(a, alpha) = Trans_x(a) * Rot_x(alpha)
    """
    rot = Rotation.from_euler('x', alpha)
    R = rot.as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[0, 3] = a
    return T


def ScrewZ(d, theta):
    """
    Screw motion along Z-axis: Translation by 'd' along Z, then rotation by 'theta' about Z.

    ScrewZ(d, theta) = Trans_z(d, theta)
    """
    rot = Rotation.from_euler('z', theta)
    R = rot.as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[2, 3] = d
    return T


def ScrewX_inv(a, alpha):
    """
    Inverse of ScrewX: ScrewX^(-1)(a, alpha) = ScrewX(-a, -alpha)
    """
    return ScrewX(-a, -alpha)


def ScrewZ_inv(d, theta):
    """
    Inverse of ScrewZ: ScrewZ^(-1)(d, theta) = ScrewZ(-d, -theta)
    """
    return ScrewZ(-d, -theta)


def dh_matrix(alpha, a, d, theta):
    """
    Create DH transformation matrix as product of screw motions.

    DH matrix = ScrewX(a, alpha) * ScrewZ(d, theta)
    """
    return ScrewX(a, alpha) @ ScrewZ(d, theta)


def dh_matrix_inverse(alpha, a, d, theta):
    """
    Compute analytical inverse of DH matrix using screw motion decomposition.

    Since DH = ScrewX(a, alpha) * ScrewZ(d, theta)
    DH^(-1) = ScrewZ^(-1)(d, theta) * ScrewX^(-1)(a, alpha)
    """
    return ScrewZ_inv(d, theta) @ ScrewX_inv(a, alpha)

def feasible_theta2_bands(a1, a2, b1, b2, c):
    """
    Compute feasible intervals (bands) for theta2 such that 
    there exists at least one solution theta1.
    Returns a list of tuples (lo, hi) representing feasible ranges in radians.
    """
    R1, R2 = np.hypot(a1, a2), np.hypot(b1, b2)
    phi2 = np.arctan2(b2, b1)

    if R2 == 0:
        # If b1 = b2 = 0, theta2 does not affect the equation
        return [(-np.pi, np.pi)], phi2

    # Range for cos(theta2 - phi2)
    L = (c - R1) / R2
    U = (c + R1) / R2

    # Clip into [-1, 1] since cos cannot go beyond that
    Lp, Up = max(-1.0, L), min(1.0, U)
    if Lp > Up:
        # No feasible interval for theta2
        return [], phi2

    alpha = np.arccos(Up)
    beta = np.arccos(Lp)

    # Two symmetric feasible intervals within one 2π cycle
    bands = [
        (phi2 + alpha, phi2 + beta),
        (phi2 - beta, phi2 - alpha)
    ]
    return bands, phi2


def pick_theta2(a1, a2, b1, b2, c):
    """
    Pick a numerically stable theta2 value: 
    the midpoint of the first feasible band.
    """
    bands, _ = feasible_theta2_bands(a1, a2, b1, b2, c)
    if not bands:
        return None
    lo, hi = bands[0]
    return (lo + hi) / 2.0


def solve_theta1_given_theta2(a1, a2, b1, b2, c, theta2):
    """
    Solve for theta1 given a chosen theta2.
    Returns two possible solutions (since cos has ± solutions).
    """
    R1 = np.hypot(a1, a2)
    R2 = np.hypot(b1, b2)
    phi1 = np.arctan2(a2, a1)
    phi2 = np.arctan2(b2, b1)

    if R1 == 0:
        # If a1 = a2 = 0, check consistency only
        ok = abs(c - R2 * np.cos(theta2 - phi2)) <= 1e-12
        return [] if not ok else [float('nan')]

    rhs = (c - R2 * np.cos(theta2 - phi2)) / R1
    rhs = np.clip(rhs, -1.0, 1.0)  # numerical safety
    t = np.arccos(rhs)

    return [phi1 + t, phi1 - t]  # plus 2k*pi for other solutions

def solve_trig_eq(a: float, b: float, c: float, verbose: bool = False) -> tuple:
    """
    Solve the trigonometric equation: a*cos(θ) + b*sin(θ) + c = 0

    Uses Weierstrass substitution: t = tan(θ/2)
    cos(θ) = (1-t**2)/(1+t**2), sin(θ) = 2t/(1+t**2)

    Returns a tuple: (real_solutions, is_arbitrary, imag_parts)
    - real_solutions: list of real parts of θ (floats, normalized to [-pi, pi])
    - is_arbitrary: True when a=b=c=0 (θ arbitrary)
    - imag_parts: list of imaginary parts for each returned root (0.0 for real roots)
    """
    tol = 1e-12
    solutions = []
    imag_parts = []
    is_arbitrary = False

    if verbose:
        print(f"Solving: {a:.6f}*cos(θ) + {b:.6f}*sin(θ) + {c:.6f} = 0")

    # Handle degenerate case: a=b=0
    if abs(a) < tol and abs(b) < tol:
        if abs(c) < tol:
            if verbose:
                print("All coefficients zero - θ is arbitrary")
            is_arbitrary = True
            return [0.0], is_arbitrary, [0.0]
        else:
            if verbose:
                print("No solutions - inconsistent equation")
            return [], is_arbitrary, []

    # Handle linear-in-sin case: a ~ 0
    if abs(a) < tol:
        if abs(b) < tol:
            return [], is_arbitrary, []
        sin_theta = -c / b
        if abs(sin_theta) <= 1.0 + 1e-12:
            sin_theta_clamped = max(-1.0, min(1.0, sin_theta))
            theta1 = math.asin(sin_theta_clamped)
            theta2 = math.pi - theta1
            solutions.extend([theta1, theta2])
            imag_parts.extend([0.0, 0.0])
            if verbose:
                print(f"Linear case in sin: θ = {theta1:.6f}, {theta2:.6f}")
        return solutions, is_arbitrary, imag_parts

    # Handle linear-in-cos case: b ~ 0
    if abs(b) < tol:
        cos_theta = -c / a
        if abs(cos_theta) <= 1.0 + 1e-12:
            cos_theta_clamped = max(-1.0, min(1.0, cos_theta))
            theta1 = math.acos(cos_theta_clamped)
            theta2 = -theta1
            solutions.extend([theta1, theta2])
            imag_parts.extend([0.0, 0.0])
            if verbose:
                print(f"Linear case in cos: θ = {theta1:.6f}, {theta2:.6f}")
        return solutions, is_arbitrary, imag_parts

    # General case: Weierstrass substitution leads to quadratic in t
    A_coeff = c - a
    B_coeff = 2.0 * b
    C_coeff = a + c

    if verbose:
        print(f"Weierstrass substitution gives quadratic: {A_coeff:.6f}*t^2 + {B_coeff:.6f}*t + {C_coeff:.6f} = 0")

    # Linear-in-t case
    if abs(A_coeff) < tol:
        if abs(B_coeff) < tol:
            if abs(C_coeff) < tol:
                if verbose:
                    print("Identity equation - infinite solutions")
                return [0.0], is_arbitrary, [0.0]
            else:
                if verbose:
                    print("Inconsistent linear equation in t")
                return [], is_arbitrary, []
        t = -C_coeff / B_coeff
        theta = 2.0 * math.atan(t)
        solutions.append(theta)
        imag_parts.append(0.0)
        if verbose:
            print(f"Linear case: t = {t:.6f}, θ = {theta:.6f}")
    else:
        # Quadratic in t: compute possibly complex roots
        discriminant = B_coeff * B_coeff - 4.0 * A_coeff * C_coeff
        if verbose:
            print(f"Discriminant = {discriminant:.6f}")
        import cmath
        sqrt_disc = cmath.sqrt(discriminant)
        t1 = (-B_coeff + sqrt_disc) / (2.0 * A_coeff)
        t2 = (-B_coeff - sqrt_disc) / (2.0 * A_coeff)

        theta1_c = 2.0 * cmath.atan(t1)
        theta2_c = 2.0 * cmath.atan(t2)

        real1, imag1 = float(theta1_c.real), float(theta1_c.imag)
        real2, imag2 = float(theta2_c.real), float(theta2_c.imag)

        solutions.extend([real1, real2])
        imag_parts.extend([imag1, imag2])

        if verbose:
            if abs(imag1) < 1e-12 and abs(imag2) < 1e-12:
                print(f"Quadratic solutions (real): t1 = {t1.real:.6f} -> θ1 = {real1:.6f}")
                print(f"                          t2 = {t2.real:.6f} -> θ2 = {real2:.6f}")
            else:
                print(f"Quadratic solutions (complex): t1 = {t1} -> θ1 = {theta1_c}")
                print(f"                             t2 = {t2} -> θ2 = {theta2_c}")

    # Normalize real parts to [-pi, pi]
    normalized_solutions = []
    for th in solutions:
        thf = float(th)
        while thf > math.pi:
            thf -= 2.0 * math.pi
        while thf < -math.pi:
            thf += 2.0 * math.pi
        normalized_solutions.append(thf)

    # Pad imag_parts if needed
    if len(imag_parts) < len(normalized_solutions):
        imag_parts.extend([0.0] * (len(normalized_solutions) - len(imag_parts)))

    return normalized_solutions, is_arbitrary, imag_parts


def solve_one_equation_for_two_unknowns(a1: float, a2: float, b1: float, b2: float, c: float,
                                      verbose: bool = False) -> list:
    """
    Solves the equation: a1*cos(θ₁) + a2*sin(θ₁) + b1*cos(θ₂) + b2*sin(θ₂) = c

    This reduces to solving one equation with two unknowns, treating one as a free parameter.
    """
    solutions = []
    
    if verbose:
        print("Solving one equation for two unknowns algebraically")
        print(f"Equation: {a1:.6f}*cos(θ₁) + {a2:.6f}*sin(θ₁) + {b1:.6f}*cos(θ₂) + {b2:.6f}*sin(θ₂) = {c:.6f}")
    
    # Check if all coefficients are zero (shouldn't happen if called correctly)
    if abs(a1) < 1e-12 and abs(a2) < 1e-12 and abs(b1) < 1e-12 and abs(b2) < 1e-12:
        if abs(c) < 1e-12:
            if verbose:
                print("Equation is 0=0, all (θ₁,θ₂) are solutions")
            # Return some representative solutions
            for i in range(1):  # Return a few solutions
                th1 = 0
                th2 = 0
                solutions.append({
                    'th1': th1,
                    'th2': th2,
                    'cos_th1': math.cos(th1),
                    'sin_th1': math.sin(th1),
                    'cos_th2': math.cos(th2),
                    'sin_th2': math.sin(th2),
                    'note': 'Trivial equation, free parameters'
                })
        else:
            if verbose:
                print("Equation is 0 = non-zero, no solutions")
        return solutions
    
    # Use pick_theta2 to choose a numerically stable theta2 value
    th2 = pick_theta2(a1, a2, b1, b2, c)

    if th2 is None:
        if verbose:
            print("No feasible theta2 values found")
        return solutions
    
    if verbose:
        print(f"Chosen theta2: {th2:.6f}")
    
    # Solve for theta1 given the chosen theta2
    th1_solutions = solve_theta1_given_theta2(a1, a2, b1, b2, c, th2)
    
    if verbose:
        print(f"Found {len(th1_solutions)} theta1 solutions for theta2={th2:.6f}")
    
    # Create solution dictionaries
    for th1 in th1_solutions:
        if not math.isnan(th1):  # Skip NaN values (which indicate consistency-only solutions)
            solutions.append({
                'th1': th1,
                'th2': th2,
                'cos_th1': math.cos(th1),
                'sin_th1': math.sin(th1),
                'cos_th2': math.cos(th2),
                'sin_th2': math.sin(th2),
                'note': 'One equation for two unknowns (θ₂ fixed)'
            })
    
    if verbose:
        print(f"Total solutions found: {len(solutions)}")
    
    return solutions


def solve_trig_sys_single(A: np.ndarray, C: np.ndarray, verbose: bool = False) -> list:
    """
    Solve the trigonometric system A[cos(x), sin(x)] = C for a single variable x.
    
    This solves systems of the form:
    A[cos(θ), sin(θ)] = C
    
    Args:
        A: 2x2 numpy array
        C: 2x1 numpy array  
        verbose: Print detailed steps
        
    Returns:
        List of solution dictionaries with 'th', 'cos_th', 'sin_th'
    """
    solutions = []
    
    if verbose:
        print(f"Solving A[cos(x), sin(x)] = C")
        print(f"A = \n{A}")
        print(f"C = {C}")
    
    # First try to invert A directly
    det_A = np.linalg.det(A)
    if abs(det_A) >= 1e-12:
        # A is invertible, solve directly
        try:
            trig_values = np.linalg.solve(A, C)
            cos_x, sin_x = trig_values[0], trig_values[1]
            
            if verbose:
                print(f"Direct solution: cos(x) = {cos_x:.6f}, sin(x) = {sin_x:.6f}")
            
            # Check trigonometric identity
            identity_error = cos_x**2 + sin_x**2 - 1
            if abs(identity_error) > 1e-4:
                if verbose:
                    print(f"Trigonometric identity violated: cos²(x) + sin²(x) - 1 = {identity_error:.2e}")
                    print("Direct solution invalid")
            else:
                # Valid solution
                x = math.atan2(sin_x, cos_x)
                solutions.append({
                    'th': x,
                    'cos_th': cos_x,
                    'sin_th': sin_x
                })
                if verbose:
                    print(f"✓ Valid solution: x = {x:.6f} rad ({math.degrees(x):.1f}°)")
                return solutions
                
        except np.linalg.LinAlgError:
            if verbose:
                print("Failed to solve directly")
    
    # A is singular or direct solution failed, use trigonometric equation approach
    if verbose:
        print("Using trigonometric equation approach")
    
    # Choose the equation with the largest coefficient magnitude
    eq_norms = []
    for i in range(2):
        eq_norm = np.linalg.norm(A[i, :])
        eq_norms.append(eq_norm)
    
    best_eq = np.argmax(eq_norms)
    if verbose:
        print(f"Using equation {best_eq} (norm = {eq_norms[best_eq]:.6f})")
    
    # Extract coefficients: A[best_eq, 0]*cos(x) + A[best_eq, 1]*sin(x) = C[best_eq]
    # Rearrange to: A[best_eq, 0]*cos(x) + A[best_eq, 1]*sin(x) - C[best_eq] = 0
    a = A[best_eq, 0]
    b = A[best_eq, 1] 
    c = -C[best_eq]  # Note: solve_trig_eq solves a*cos + b*sin + c = 0
    
    if verbose:
        print(f"Solving: {a:.6f}*cos(x) + {b:.6f}*sin(x) + {c:.6f} = 0")
    
    # Solve using the trigonometric equation solver
    x_solutions, _, _ = solve_trig_eq(a, b, c, verbose)
    
    for x in x_solutions:
        cos_x = math.cos(x)
        sin_x = math.sin(x)
        
        # Verify the solution satisfies the original system
        residual = A @ np.array([cos_x, sin_x]) - C
        if np.linalg.norm(residual) < 1e-6:
            solutions.append({
                'th': x,
                'cos_th': cos_x,
                'sin_th': sin_x
            })
            if verbose:
                print(f"✓ Valid solution: x = {x:.6f} rad ({math.degrees(x):.1f}°)")
    
    return solutions


def pre_process(A: np.ndarray, B: np.ndarray, C: np.ndarray, verbose: bool = False) -> tuple:
    """
    Preprocess the trigonometric system matrices A, B, C.
    
    Performs normalization and row reduction to improve numerical stability.
    
    Args:
        A: 2x2 numpy array
        B: 2x2 numpy array
        C: 2x1 numpy array
        verbose: Print detailed steps
        
    Returns:
        Tuple of (rank_A, rank_B, A_norm, B_norm, C_norm)
    """
    # Compute ranks on original matrices
    rank_A = np.linalg.matrix_rank(A, tol=1e-12)
    rank_B = np.linalg.matrix_rank(B, tol=1e-12)
    
    # Create copies for preprocessing
    A_norm = A.copy()
    B_norm = B.copy()
    C_norm = C.copy()
    
    # Track which equations are trivial (0=0)
    trivial_equations = [False, False]
    
    if verbose:
        print("\nPreprocessing:")
    
    # First, check for trivial equations and move to second row if found
    for i in range(2):
        coeff_vector = np.array([A[i, 0], A[i, 1], B[i, 0], B[i, 1]])
        coeff_norm = np.linalg.norm(coeff_vector)
        if coeff_norm < 1e-12 and abs(C[i]) < 1e-12:
            trivial_equations[i] = True
    
    # If there's a trivial equation in row 0, move it to row 1
    if trivial_equations[0]:
        A_norm[[0, 1]] = A_norm[[1, 0]]
        B_norm[[0, 1]] = B_norm[[1, 0]]
        C_norm[[0, 1]] = C_norm[[1, 0]]
        trivial_equations[0], trivial_equations[1] = trivial_equations[1], trivial_equations[0]
        if verbose:
            print("  Moved trivial equation to row 1")
    
    # If there is any trivial equation, return without scaling
    if any(trivial_equations):
        return rank_A, rank_B, A_norm, B_norm, C_norm
    
    # Step 1: Normalize each equation by the norm of its coefficients
    for i in range(2):
        # Compute norm of all coefficients for equation i: [A[i,0], A[i,1], B[i,0], B[i,1]]
        coeff_vector = np.array([A[i, 0], A[i, 1], B[i, 0], B[i, 1]])
        coeff_norm = np.linalg.norm(coeff_vector)
        
        if coeff_norm < 1e-12:
            # Equation is 0 = C[i], check if it's 0=0
            if abs(C[i]) < 1e-12:
                trivial_equations[i] = True
                if verbose:
                    print(f"  Equation {i+1}: 0 = 0 (trivial, all coefficients zero)")
            else:
                if verbose:
                    print(f"  Equation {i+1}: 0 = {C[i]:.6f} (inconsistent)")
        else:
            # Normalize the equation
            A_norm[i, :] /= coeff_norm
            B_norm[i, :] /= coeff_norm
            C_norm[i] /= coeff_norm
            if verbose:
                print(f"  Equation {i+1}: normalized by factor {coeff_norm:.6f}")
    
    # Step 2: Row reduction to eliminate B[1,0] (or A[1,0] if B is zero matrix)
    if not trivial_equations[0] and not trivial_equations[1]:
        # Check if B is effectively zero matrix
        B_norm_max = np.max(np.abs(B_norm))
        if B_norm_max < 1e-12:
            # B is zero matrix, eliminate A[1,0] to make the system easier to solve
            if abs(A_norm[0, 0]) > 1e-12:
                # Use row 0 to eliminate A[1,0]
                factor = A_norm[1, 0] / A_norm[0, 0]
                A_norm[1, :] -= factor * A_norm[0, :]
                C_norm[1] -= factor * C_norm[0]
                if verbose:
                    print(f"  Row reduction: eliminated A[1,0] using factor {factor:.6f}")
        else:
            # Normal case: eliminate B[1,0] to simplify the system
            if abs(B_norm[0, 0]) <= 1e-12:
                # Check if second row already has zero B coefficients
                if abs(B_norm[1, 0]) <= 1e-12 and abs(B_norm[1, 1]) <= 1e-12:
                    # Second row is already [0, 0] for B coefficients, no swap needed
                    if verbose:
                        print("  No row swap needed: second row already has zero B coefficients")
                else:
                    # Swap rows to bring a non-zero B[0,0] to the first row
                    A_norm[[0, 1]] = A_norm[[1, 0]]
                    B_norm[[0, 1]] = B_norm[[1, 0]]
                    C_norm[[0, 1]] = C_norm[[1, 0]]
                    if verbose:
                        print("  Row swap: swapped rows to get non-zero B[0,0]")
            else:
                # Use row 0 to eliminate B[1,0]
                factor = B_norm[1, 0] / B_norm[0, 0]
                A_norm[1, :] -= factor * A_norm[0, :]
                B_norm[1, :] -= factor * B_norm[0, :]
                C_norm[1] -= factor * C_norm[0]
                if verbose:
                    print(f"  Row reduction: eliminated B[1,0] using factor {factor:.6f}")
    
    if verbose:
        print("  After preprocessing:")
        print(f"  A_norm = \n{A_norm}")
        print(f"  B_norm = \n{B_norm}")
        print(f"  C_norm = {C_norm}")
    
    # CRITICAL FIX: Recompute ranks AFTER preprocessing
    # The row reduction can change the ranks (especially B can go from rank-2 to rank-1)
    rank_A_final = np.linalg.matrix_rank(A_norm, tol=1e-12)
    rank_B_final = np.linalg.matrix_rank(B_norm, tol=1e-12)
    
    if verbose and (rank_A_final != rank_A or rank_B_final != rank_B):
        print(f"  Rank changed after preprocessing: rank_A {rank_A} -> {rank_A_final}, rank_B {rank_B} -> {rank_B_final}")
    
    return rank_A_final, rank_B_final, A_norm, B_norm, C_norm

def _deduplicate_angle_pairs(pairs: List[Tuple[float, float]], 
                             tol: float = 1e-10) -> List[Tuple[float, float]]:
    """Remove duplicate angle pairs within tolerance."""
    if not pairs:
        return []
    
    unique = [pairs[0]]
    for pair in pairs[1:]:
        is_duplicate = False
        for u_pair in unique:
            dist = np.sqrt((pair[0] - u_pair[0])**2 + (pair[1] - u_pair[1])**2)
            if dist < tol:
                is_duplicate = True
                break
        if not is_duplicate:
            unique.append(pair)
    return unique

def _deduplicate_values(values: List[float], tol: float = 1e-10) -> List[float]:
    """Remove duplicate numeric values within tolerance."""
    if not values:
        return []
    
    unique = [values[0]]
    for val in values[1:]:
        if not any(abs(val - u) < tol for u in unique):
            unique.append(val)
    return unique

def solve_bilinear_two_angles_numeric_legacy(coeffs1: List[float], coeffs2: List[float], 
                                             principal_range: bool = True, 
                                             tol: float = 1e-9,
                                             max_roots: int = 64,
                                             verbose: bool = False) -> List[Tuple[float, float]]:
    """
    Legacy numeric solver for the bilinear two-angle system using
    tangent-half substitution, Sylvester resultant, and an eigenvalue solver.
    """
    # Extract coefficients
    k10, k11c, k11s, k12c, k12s, k1cc, k1cs, k1sc, k1ss = coeffs1
    k20, k21c, k21s, k22c, k22s, k2cc, k2cs, k2sc, k2ss = coeffs2
    
    # After tan-half substitution: t1 = tan(th1/2), t2 = tan(th2/2)
    # Cos[th1] = (1-t1²)/(1+t1²), Sin[th1] = 2*t1/(1+t1²)
    # Cos[th2] = (1-t2²)/(1+t2²), Sin[th2] = 2*t2/(1+t2²)
    # Clear denominators (1+t1²)(1+t2²)
    
    # Build coefficient matrices for polynomials in t1 and t2
    # p1(t1,t2) and p2(t1,t2) are degree 2 in both variables
    # p1_coeffs[i][j] = coefficient of t1^i * t2^j
    
    p1_coeffs = np.array([
        [k10 + k11c + k12c + k1cc,  2.0*(k12s + k1cs),  k10 + k11c - k12c - k1cc],
        [2.0*(k11s + k1sc),         4.0*k1ss,           2.0*(k11s - k1sc)],
        [k10 - k11c + k12c - k1cc,  2.0*(k12s - k1cs),  k10 - k11c - k12c + k1cc]
    ])
    
    p2_coeffs = np.array([
        [k20 + k21c + k22c + k2cc,  2.0*(k22s + k2cs),  k20 + k21c - k22c - k2cc],
        [2.0*(k21s + k2sc),         4.0*k2ss,           2.0*(k21s - k2sc)],
        [k20 - k21c + k22c - k2cc,  2.0*(k22s - k2cs),  k20 - k21c - k22c + k2cc]
    ])
    
    # Build Sylvester resultant matrix (4x4 for degree-2 polynomials in t2)
    # Sylvester matrix eliminates t2, leaving polynomial in t1
    # Each element of Sylvester matrix is a polynomial in t1
    
    # Sylvester matrix structure (4x4):
    # Row 1: [p1[t2²], p1[t2¹], p1[t2⁰], 0]
    # Row 2: [0, p1[t2²], p1[t2¹], p1[t2⁰]]
    # Row 3: [p2[t2²], p2[t2¹], p2[t2⁰], 0]
    # Row 4: [0, p2[t2²], p2[t2¹], p2[t2⁰]]
    
    # Each p[t2^k] is a polynomial in t1: sum over i of p_coeffs[i, 2-k] * t1^i
    
    # Build coefficient matrices for Sylvester matrix polynomial in t1
    # coeffMatrices[k] = 4x4 matrix of coefficients of t1^k in Sylvester matrix
    coeff_matrices = []
    for k in range(3):  # k = 0, 1, 2 (degree of t1)
        sylv_k = np.array([
            [p1_coeffs[k, 2], p1_coeffs[k, 1], p1_coeffs[k, 0], 0.0],
            [0.0, p1_coeffs[k, 2], p1_coeffs[k, 1], p1_coeffs[k, 0]],
            [p2_coeffs[k, 2], p2_coeffs[k, 1], p2_coeffs[k, 0], 0.0],
            [0.0, p2_coeffs[k, 2], p2_coeffs[k, 1], p2_coeffs[k, 0]]
        ])
        coeff_matrices.append(sylv_k)
    
    # Linearize into companion matrix form and solve eigenvalue problem
    # Matrix polynomial: M0 + M1*λ + M2*λ² = 0
    # Correct companion form: A = [[0, I], [-M2^-1@M0, -M2^-1@M1]]
    # Then solve A*v = λ*v for eigenvalues λ (roots in t1)
    
    M0, M1, M2 = coeff_matrices[0], coeff_matrices[1], coeff_matrices[2]
    n = M0.shape[0]
    
    # Check singularity of M2 using both determinant and condition number
    # (more robust than catching exception after inversion)
    det_M2 = np.linalg.det(M2)
    det_threshold = 1e-10  # Threshold for singularity detection
    
    # Also check condition number to detect near-singular matrices
    try:
        cond_M2 = np.linalg.cond(M2)
        cond_threshold = 1e12  # Threshold for ill-conditioned matrix
    except:
        cond_M2 = float('inf')
    
    # Use generalized eigenvalue if M2 is singular or ill-conditioned
    use_generalized = (abs(det_M2) <= det_threshold) or (cond_M2 >= cond_threshold)
    
    eigs = None
    eigvecs = None
    
    if not use_generalized:
        # M2 is well-conditioned, use standard companion matrix approach
        try:
            M2_inv = np.linalg.inv(M2)
            # Check if inversion produced reasonable results (no huge numbers)
            max_inv_entry = np.max(np.abs(M2_inv))
            if max_inv_entry > 1e12:
                # Inversion produced huge numbers, fall back to generalized eigenvalue
                use_generalized = True
            else:
                companion = np.block([[np.zeros((n, n)), np.eye(n)],
                                      [-M2_inv @ M0, -M2_inv @ M1]])
                eigs, eigvecs = np.linalg.eig(companion)
        except (np.linalg.LinAlgError, ValueError):
            # Inversion failed, use generalized eigenvalue
            use_generalized = True
    
    if use_generalized:
        # M2 is singular/ill-conditioned, use generalized eigenvalue formulation
        # For M0 + M1*λ + M2*λ² = 0, solve A*v = λ*B*v where:
        # A = [[-M1, -M0], [I, 0]], B = [[M2, 0], [0, I]]
        try:
            A_block = np.block([[-M1, -M0], [np.eye(n), np.zeros((n, n))]])
            B_block = np.block([[M2, np.zeros((n, n))], [np.zeros((n, n)), np.eye(n)]])
            from scipy.linalg import eig as scipy_eig
            eigs, eigvecs = scipy_eig(A_block, B_block)
        except:
            return []
    
    # Filter eigenvalues (roots of t1 = tan(th1/2))
    # Treat very large eigenvalues as "effectively infinite" (corresponding to ±180°)
    # Also extract corresponding eigenvectors to get t2 values
    roots1 = []
    eigvec_map = {}  # Map from t1 to its eigenvector
    large_threshold = 1e6  # Threshold for "effectively infinite"
    
    if verbose:
        print(f"Eigenvalues: {eigs}")
        finite_eigs = [e for e in eigs if np.isfinite(e)]
        infinite_eigs = [e for e in eigs if not np.isfinite(e)]
        large_eigs = [e for e in finite_eigs if abs(e) >= large_threshold]
        print(f"Finite eigenvalues: {len(finite_eigs)}")
        print(f"Infinite eigenvalues: {infinite_eigs}")
        print(f"Very large eigenvalues (|λ| ≥ {large_threshold}): {large_eigs}")
    
    for i, eig in enumerate(eigs):
        if not np.isfinite(eig):
            # Truly infinite eigenvalue (inf or nan)
            if verbose:
                print(f"Found infinite eigenvalue: {eig}, interpreting as θ = ±180°")
            # Extract t2 from eigenvector
            # For generalized eigenvalue: eigenvector is [t2^(n-1), ..., t2, 1, t1*t2^(n-1), ..., t1*t2, t1]
            # The last component is t1 (→∞), second-to-last is t1*t2, so t2 = (second-to-last)/(last)
            eigvec = eigvecs[:, i]
            # For 2D system: eigvec has components [t2, 1, t1*t2, t1]
            # t2 is at index 0, t1 is at index 3
            if abs(eigvec[3]) > tol:
                t2_from_eigvec = eigvec[2] / eigvec[3]  # (t1*t2) / t1
                if verbose:
                    print(f"  Eigenvector: {eigvec}")
                    print(f"  t2 from eigenvector: {t2_from_eigvec}")
                eigvec_map[1e10] = t2_from_eigvec
                eigvec_map[-1e10] = t2_from_eigvec
            # Interpret as ±180° - add both candidates
            roots1.append(1e10)   # Represents t → +∞ (th → +π)
            roots1.append(-1e10)  # Represents t → -∞ (th → -π)
        elif abs(np.imag(eig)) < tol:
            # Real eigenvalue (possibly very large)
            real_eig = np.real(eig)
            eigvec = eigvecs[:, i]
            
            if abs(real_eig) >= large_threshold:
                # Very large eigenvalue - treat as effectively infinite
                if verbose:
                    print(f"Found very large eigenvalue: {real_eig:.2e}, interpreting as θ ≈ ±180°")
                    print(f"  Eigenvector: {eigvec}")
                    print(f"  eigvec[3] = {eigvec[3]}, abs = {abs(eigvec[3])}")
                # Extract t2 from eigenvector
                # For large eigenvalues, eigvec[3] can be very small (e.g., 1e-10)
                # Use eigvec relative to its norm instead of absolute tolerance
                eigvec_norm = np.linalg.norm(eigvec)
                if abs(eigvec[3]) / eigvec_norm > 1e-12:  # Relative tolerance
                    t2_from_eigvec = eigvec[2] / eigvec[3]  # (t1*t2) / t1
                    eigvec_map[real_eig] = t2_from_eigvec
                    if verbose:
                        print(f"  ✓ t2 from eigenvector: {t2_from_eigvec}")
                elif verbose:
                    print(f"  ✗ eigvec[3] too small relative to norm ({abs(eigvec[3])/eigvec_norm:.2e}), cannot extract t2")
                # Use the large value directly (represents t → ±∞)
                roots1.append(real_eig)
            else:
                # Normal finite eigenvalue
                # Also store eigenvector for t2 extraction
                if abs(eigvec[3]) > tol:
                    t2_from_eigvec = eigvec[2] / eigvec[3]
                    eigvec_map[real_eig] = t2_from_eigvec
                roots1.append(real_eig)
    
    roots1 = _deduplicate_values(roots1, tol)
    roots1 = roots1[:max_roots]
    
    if verbose:
        print(f"roots1 (t1 values including ±∞): {roots1}")
    
    # For each t1 root, solve for t2
    solutions = []
    for t1_val in roots1:
        # Check if we have eigenvector information for this t1
        if t1_val in eigvec_map:
            # Use t2 from eigenvector; if complex with small imaginary part keep real part
            t2_from_eigvec = eigvec_map[t1_val]
            if isinstance(t2_from_eigvec, complex):
                if abs(t2_from_eigvec.imag) < 1e-8:
                    t2_clean = float(t2_from_eigvec.real)
                else:
                    # Record both real projection and magnitude-based guess; legacy solver was over restrictive
                    t2_clean = float(t2_from_eigvec.real)
            else:
                t2_clean = float(t2_from_eigvec)
            roots2 = [t2_clean]
            if verbose:
                print(f"Using t2 from eigenvector for t1 = {t1_val:.2e}")
                print(f"  raw t2 = {t2_from_eigvec}, cleaned t2 = {t2_clean}")
        # Handle large t1 values (representing th1 → ±180°)
        # When |t1| is very large, use limiting behavior of polynomial
        elif abs(t1_val) > 1e8:
            # For large t1, polynomial ~ t1² * (leading coefficient in t1)
            # p(t1,t2) ≈ t1² * p_coeffs[2, :]
            p1_t2 = p1_coeffs[2, :]  # Coefficients of t2^0, t2^1, t2^2
            p2_t2 = p2_coeffs[2, :]
            
            # ALSO: Add t2=0 and t2=±∞ as explicit candidates for th2
            # This handles cases where th1=±180° pairs with th2=0° or ±180°
            roots2 = [0.0, 1e10, -1e10]  # th2 = 0°, ±180°
            
            if verbose:
                print(f"Using limiting form for large t1 = {t1_val:.2e}")
                print(f"  p1_t2 (limiting): {p1_t2}")
                print(f"  p2_t2 (limiting): {p2_t2}")
                print(f"  Adding explicit candidates: t2 = 0, ±∞")
            
            # Skip the polynomial solving for this t1 value
            # Jump directly to verification with explicit candidates
        else:
            # Normal case: evaluate polynomial at t1 = t1_val
            p1_t2 = np.array([
                p1_coeffs[0, 0] + p1_coeffs[1, 0]*t1_val + p1_coeffs[2, 0]*t1_val**2,
                p1_coeffs[0, 1] + p1_coeffs[1, 1]*t1_val + p1_coeffs[2, 1]*t1_val**2,
                p1_coeffs[0, 2] + p1_coeffs[1, 2]*t1_val + p1_coeffs[2, 2]*t1_val**2
            ])
            
            p2_t2 = np.array([
                p2_coeffs[0, 0] + p2_coeffs[1, 0]*t1_val + p2_coeffs[2, 0]*t1_val**2,
                p2_coeffs[0, 1] + p2_coeffs[1, 1]*t1_val + p2_coeffs[2, 1]*t1_val**2,
                p2_coeffs[0, 2] + p2_coeffs[1, 2]*t1_val + p2_coeffs[2, 2]*t1_val**2
            ])
            
            # Solve quadratics: p1_t2[2]*t2² + p1_t2[1]*t2 + p1_t2[0] = 0
            roots2 = []
            
            if verbose:
                print(f"Solving for t2 given t1 = {t1_val:.2e}")
            
            # Solve p1
            if abs(p1_t2[2]) > tol:
                disc = p1_t2[1]**2 - 4*p1_t2[2]*p1_t2[0]
                if disc >= -tol:
                    disc = max(disc, 0.0)
                    roots2.append((-p1_t2[1] + np.sqrt(disc)) / (2*p1_t2[2]))
                    roots2.append((-p1_t2[1] - np.sqrt(disc)) / (2*p1_t2[2]))
            
            # Solve p2
            if abs(p2_t2[2]) > tol:
                disc = p2_t2[1]**2 - 4*p2_t2[2]*p2_t2[0]
                if disc >= -tol:
                    disc = max(disc, 0.0)
                    roots2.append((-p2_t2[1] + np.sqrt(disc)) / (2*p2_t2[2]))
                    roots2.append((-p2_t2[1] - np.sqrt(disc)) / (2*p2_t2[2]))
            
            roots2 = _deduplicate_values(roots2, tol)
        
        if verbose and abs(t1_val) > 1e8:
            print(f"  Found {len(roots2)} t2 roots: {roots2}")
        
        # Convert t values to angles and verify
        for t2_val in roots2:
            # Handle large t values (representing th → ±180°)
            # When |t| is very large, tan(th/2) → ±∞, so th → ±π
            if abs(t1_val) > 1e8:
                th1 = np.pi if t1_val > 0 else -np.pi
            else:
                th1 = 2.0 * np.arctan(t1_val)
            
            if abs(t2_val) > 1e8:
                th2 = np.pi if t2_val > 0 else -np.pi
            else:
                th2 = 2.0 * np.arctan(t2_val)
            
            # Verify solution
            c1, s1 = np.cos(th1), np.sin(th1)
            c2, s2 = np.cos(th2), np.sin(th2)
            
            res1 = (k10 + k11c*c1 + k11s*s1 + k12c*c2 + k12s*s2 +
                   k1cc*c1*c2 + k1cs*c1*s2 + k1sc*s1*c2 + k1ss*s1*s2)
            res2 = (k20 + k21c*c1 + k21s*s1 + k22c*c2 + k22s*s2 +
                   k2cc*c1*c2 + k2cs*c1*s2 + k2sc*s1*c2 + k2ss*s1*s2)
            
            max_res = max(abs(res1), abs(res2))
            
            if verbose and abs(t1_val) > 1e8:
                print(f"  Candidate: th1={np.rad2deg(th1):.1f}°, th2={np.rad2deg(th2):.1f}°, residual={max_res:.2e}")
            
            # For large eigenvalues (near ±180°), use relaxed tolerance due to numerical precision
            effective_tol = 1e-4 if (abs(t1_val) > 1e6 or abs(t2_val) > 1e6) else tol
            
            if max_res < effective_tol:
                if principal_range:
                    th1 = np.arctan2(np.sin(th1), np.cos(th1))
                    th2 = np.arctan2(np.sin(th2), np.cos(th2))
                solutions.append((th1, th2))
    
    return _deduplicate_angle_pairs(solutions, tol)


def solve_bilinear_two_angles_numeric(coeffs1: List[float], coeffs2: List[float], 
                                      principal_range: bool = True, 
                                      tol: float = 1e-9,
                                      max_roots: int = 64,
                                      verbose: bool = False) -> List[Tuple[float, float]]:
    """Wrapper that attempts robust closed-form solver first, then falls back.

    The original implementation has been preserved as
    solve_bilinear_two_angles_numeric_legacy.
    """
    if _solve_two_angle_system is None or _EqCoeffs is None or _TwoEqSystem is None:
        if verbose:
            print("Robust solver not available; using legacy implementation.")
        return solve_bilinear_two_angles_numeric_legacy(
            coeffs1, coeffs2, principal_range=principal_range, tol=tol,
            max_roots=max_roots, verbose=verbose
        )

    # Map coefficient arrays to EquationCoefficients dataclasses
    try:
        eq1 = _EqCoeffs(
            k0=float(coeffs1[0]),
            k1c=float(coeffs1[1]),
            k1s=float(coeffs1[2]),
            k2c=float(coeffs1[3]),
            k2s=float(coeffs1[4]),
            kcc=float(coeffs1[5]),
            kcs=float(coeffs1[6]),
            ksc=float(coeffs1[7]),
            kss=float(coeffs1[8]),
        )
        eq2 = _EqCoeffs(
            k0=float(coeffs2[0]),
            k1c=float(coeffs2[1]),
            k1s=float(coeffs2[2]),
            k2c=float(coeffs2[3]),
            k2s=float(coeffs2[4]),
            kcc=float(coeffs2[5]),
            kcs=float(coeffs2[6]),
            ksc=float(coeffs2[7]),
            kss=float(coeffs2[8]),
        )
    except Exception as e:
        if verbose:
            print(f"Coefficient mapping failed ({e}); using legacy solver.")
        return solve_bilinear_two_angles_numeric_legacy(
            coeffs1, coeffs2, principal_range=principal_range, tol=tol,
            max_roots=max_roots, verbose=verbose
        )

    system = _TwoEqSystem(eq1=eq1, eq2=eq2)

    # Call the robust solver
    try:
        sols = _solve_two_angle_system(system, tol=tol)
    except Exception as e:
        if verbose:
            print(f"Robust solver raised {e}; using legacy solver.")
        return solve_bilinear_two_angles_numeric_legacy(
            coeffs1, coeffs2, principal_range=principal_range, tol=tol,
            max_roots=max_roots, verbose=verbose
        )

    # Convert to list of (q1, q2). Filter out 'free' or None angle entries.
    results: List[Tuple[float, float]] = []
    for sol in sols:
        q1 = getattr(sol, "q1", None)
        q2 = getattr(sol, "q2", None)
        free_q1 = getattr(sol, "free_q1", False)
        free_q2 = getattr(sol, "free_q2", False)
        if q1 is None or q2 is None:
            continue
        if free_q1 or free_q2:
            continue
        if principal_range:
            q1 = float(np.arctan2(np.sin(q1), np.cos(q1)))
            q2 = float(np.arctan2(np.sin(q2), np.cos(q2)))
        results.append((q1, q2))

    results = _deduplicate_angle_pairs(results, tol)

    if not results:
        if verbose:
            print("Robust solver returned no finite (q1,q2) solutions. Checking for free-angle degeneracy before fallback...")
        # If robust solver produced a free-q2 or free-q1 solution we may want to expose that as empty list instead of legacy fallback.
        produced_free = any(getattr(s, 'free_q1', False) or getattr(s, 'free_q2', False) for s in sols)
        if produced_free:
            if verbose:
                print("Detected free-angle condition; returning empty numeric list (caller may handle separately).")
            return []
        if verbose:
            print("No degeneracy detected; falling back to legacy solver.")
        return solve_bilinear_two_angles_numeric_legacy(
            coeffs1, coeffs2, principal_range=principal_range, tol=tol,
            max_roots=max_roots, verbose=verbose
        )

    return results
