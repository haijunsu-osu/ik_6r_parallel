"""
Inverse Kinematics Solver for 6R Robot with 3 Parallel Joints (joints 2, 3, 4)
DH Parameters: {{0,0,0,q1}, {α1,a1,d2,q2}, {0,a2,d3,q3}, {0,a3,d4,q4}, {α4,a4,d5,q5}, {α5,a5,0,q6}}
where α2 = α3 = 0 (joints 2, 3, 4 are parallel)

DH Table:
Joint  |  α    |  a   |  d   |  θ
-------|-------|------|------|-----
0→1    |  0    |  0   |  d1  |  q1
1→2    |  α1   |  a1  |  d2  |  q2
2→3    |  0    |  a2  |  d3  |  q3
3→4    |  0    |  a3  |  d4  |  q4
4→5    |  α4   |  a4  |  d5  |  q5
5→6    |  α5   |  a5  |  d6  |  q6

Strategy:
1. Solve for q1 and q6 from bilinear equations (Eqth161, Eqth162)
   - Decoupled case (a5=d5=0): Equations separate, solve sequentially
   - Coupled case (a5≠0 or d5≠0): Use general bilinear solver
2. Solve for q5 from orientation constraint (T5Left[[1:3,3]] = [0,0,1])
3. Solve for q2, q3, q4 from planar 3R IK

Note: The coupled case (a5≠0) is significantly more robust than the decoupled case (a5=d5=0).
Stress tests show: Coupled (a5≠0): 89% success rate, Decoupled (a5=d5=0): 22% success rate.
For maximum robustness, consider using non-zero a5 (e.g., a5=50mm) if physically feasible.

Typical solution count: 8 solutions (4 combinations of (q1,q5,q6) × 2 elbow configs for (q2,q3,q4))
"""

import numpy as np
import sys
import os


# Add path to import modules from other directories
# Ensure the repository "scripts" folder and subfolders are on sys.path so
# local modules (Spatial3R, spherical_wrist_ik_solver, utilities, etc.)
# can be imported regardless of the current working directory.
base_scripts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Prefer inserting at front so these local modules shadow installed packages if any.
sys.path.insert(0, os.path.join(base_scripts_dir, 'ik_3r_position'))
sys.path.insert(0, os.path.join(base_scripts_dir, 'ik_3r_wrist'))
sys.path.insert(0, os.path.join(base_scripts_dir, 'helpers'))
sys.path.insert(0, base_scripts_dir)

from utilities import solve_trig_eq, solve_bilinear_two_angles_numeric, solve_trig_sys_single
from numerical_solver import solve_trig_sys
from two_angle_solver import (
    TwoEquationSystem, 
    EquationCoefficients, 
    solve_bilinear_sys
)


def dh_transform(alpha, a, d, theta):
    """
    Compute the DH transformation matrix.
    T = Rot(X, alpha) * Trans(X, a) * Trans(Z, d) * Rot(Z, theta)
    """
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    
    return np.array([
        [ct, -st, 0, a],
        [st*ca, ct*ca, -sa, -sa*d],
        [st*sa, ct*sa, ca, ca*d],
        [0, 0, 0, 1]
    ])


def inv_dh_transform(alpha, a, d, theta):
    """
    Compute the inverse DH transformation matrix.
    """
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    
    # Correct formula: invDHTrans[α, a, d, θ] = ZDisp[-d, -θ] . XDisp[-a, -α]
    # Result from Mathematica:
    # {{ct, ca*st, sa*st, -a*ct},
    #  {-st, ca*ct, sa*ct, a*st},
    #  {0, -sa, ca, -d},
    #  {0, 0, 0, 1}}
    return np.array([
        [ct, ca*st, sa*st, -a*ct],
        [-st, ca*ct, sa*ct, a*st],
        [0, -sa, ca, -d],
        [0, 0, 0, 1]
    ])


def forward_kinematics(dh_params, joint_angles):
    """
    Compute forward kinematics for 6R robot using Modified DH convention.
    
    Compatible with RobotKinematicsCatalogue DH parameter format.
    
    Args:
        dh_params: dict with keys 'alpha0', 'a0', 'd1', 'alpha1', 'a1', 'd2', 
                   'alpha2', 'a2', 'd3', 'alpha3', 'a3', 'd4', 'alpha4', 'a4', 'd5', 
                   'alpha5', 'a5', 'd6'
        joint_angles: [q1, q2, q3, q4, q5, q6] (in radians)
    
    Returns:
        T06: 4x4 homogeneous transformation matrix from frame 0 to frame 6
    """
    q1, q2, q3, q4, q5, q6 = joint_angles
    
    # Modified DH transformations (RobotKinematicsCatalogue convention)
    T01 = dh_transform(dh_params['alpha0'], dh_params['a0'], dh_params['d1'], q1)
    T12 = dh_transform(dh_params['alpha1'], dh_params['a1'], dh_params['d2'], q2)
    T23 = dh_transform(dh_params['alpha2'], dh_params['a2'], dh_params['d3'], q3)
    T34 = dh_transform(dh_params['alpha3'], dh_params['a3'], dh_params['d4'], q4)
    T45 = dh_transform(dh_params['alpha4'], dh_params['a4'], dh_params['d5'], q5)
    T56 = dh_transform(dh_params['alpha5'], dh_params['a5'], dh_params['d6'], q6)
    
    T06 = T01 @ T12 @ T23 @ T34 @ T45 @ T56
    return T06


def compute_bilinear_coefficients(T06, dh_params):
    """
    Compute coefficients for the bilinear equations in q1 and q6.
    
    Equations:
    - Eqth161: z4 · z5 - cos(α4) = 0
    - Eqth162: (P05Right - P05Left) · z2 = 0
    
    Returns:
        coeffs1, coeffs2: Coefficients for the two bilinear equations
                          [k0, kc1, ks1, kc6, ks6, kcc, kcs, ksc, kss]
    """
    # Extract target pose
    R06 = T06[:3, :3]
    P06 = T06[:3, 3]
    px, py, pz = P06
    
    # Extract DH parameters
    alpha1 = dh_params['alpha1']
    a1 = dh_params['a1']
    d2 = dh_params['d2']
    a2 = dh_params['a2']
    d3 = dh_params['d3']
    a3 = dh_params['a3']
    d4 = dh_params['d4']
    alpha4 = dh_params['alpha4']
    a4 = dh_params['a4']
    d5 = dh_params['d5']
    alpha5 = dh_params['alpha5']
    a5 = dh_params['a5']
    
    ca1, sa1 = np.cos(alpha1), np.sin(alpha1)
    ca4, sa4 = np.cos(alpha4), np.sin(alpha4)
    ca5, sa5 = np.cos(alpha5), np.sin(alpha5)
    
    r11, r12, r13 = R06[0, :]
    r21, r22, r23 = R06[1, :]
    r31, r32, r33 = R06[2, :]
    
    # ========== Equation 1: z4 · z5 - cos(α4) = 0 ==========
    # From symbolic derivation in Mathematica
    # k0  = -Cos[alpha4] + r33*Cos[alpha1]*Cos[alpha5]
    # kc1 = -(r23*Cos[alpha5]*Sin[alpha1])
    # ks1 = r13*Cos[alpha5]*Sin[alpha1]
    # kc6 = r32*Cos[alpha1]*Sin[alpha5]
    # ks6 = r31*Cos[alpha1]*Sin[alpha5]
    # kcc = -(r22*Sin[alpha1]*Sin[alpha5])
    # kcs = -(r21*Sin[alpha1]*Sin[alpha5])
    # ksc = r12*Sin[alpha1]*Sin[alpha5]
    # kss = r11*Sin[alpha1]*Sin[alpha5]
    
    k0_1 = -ca4 + r33*ca1*ca5 # +/- ca4
    kc1_1 = -(r23*ca5*sa1)
    ks1_1 = r13*ca5*sa1
    kc6_1 = r32*ca1*sa5
    ks6_1 = r31*ca1*sa5
    kcc_1 = -(r22*sa1*sa5)
    kcs_1 = -(r21*sa1*sa5)
    ksc_1 = r12*sa1*sa5
    kss_1 = r11*sa1*sa5
    
    coeffs1 = [k0_1, kc1_1, ks1_1, kc6_1, ks6_1, kcc_1, kcs_1, ksc_1, kss_1]
    
    # ========== Equation 2: (P05Right - P05Left) · z2 = 0 ==========
    # From symbolic derivation in Mathematica
    # k0  = -d2 - d3 - d4 + Cos[alpha1]*(pz - d5*r33*Cos[alpha5])
    # kc1 = (-py + d5*r23*Cos[alpha5])*Sin[alpha1]
    # ks1 = (px - d5*r13*Cos[alpha5])*Sin[alpha1]
    # kc6 = -(Cos[alpha1]*(a5*r31 + d5*r32*Sin[alpha5]))
    # ks6 = Cos[alpha1]*(a5*r32 - d5*r31*Sin[alpha5])
    # kcc = Sin[alpha1]*(a5*r21 + d5*r22*Sin[alpha5])
    # kcs = Sin[alpha1]*(-(a5*r22) + d5*r21*Sin[alpha5])
    # ksc = -(Sin[alpha1]*(a5*r11 + d5*r12*Sin[alpha5]))
    # kss = Sin[alpha1]*(a5*r12 - d5*r11*Sin[alpha5])
    
    k0_2 = -d2 - d3 - d4 + pz*ca1 - d5*r33*ca1*ca5
    kc1_2 = -(py*sa1) + d5*r23*ca5*sa1
    ks1_2 = px*sa1 - d5*r13*ca5*sa1
    kc6_2 = -(a5*r31*ca1) - d5*r32*ca1*sa5
    ks6_2 = a5*r32*ca1 - d5*r31*ca1*sa5
    kcc_2 = a5*r21*sa1 + d5*r22*sa1*sa5
    kcs_2 = -(a5*r22*sa1) + d5*r21*sa1*sa5
    ksc_2 = -(a5*r11*sa1) - d5*r12*sa1*sa5
    kss_2 = a5*r12*sa1 - d5*r11*sa1*sa5
    
    coeffs2 = [k0_2, kc1_2, ks1_2, kc6_2, ks6_2, kcc_2, kcs_2, ksc_2, kss_2]
    
    return coeffs1, coeffs2


def solve_q1_q6_decoupled(T06, dh_params, verbose=False):
    """
    Solve for q1 and q6 when a5 = d5 = 0 (decoupled equations).
    
    Equation 2: -d2 - d3 - d4 + pz*cos(alpha1) - py*cos(q1)*sin(alpha1) + px*sin(alpha1)*sin(q1) = 0
    Equation 1: Complex equation involving q1 and q6
    
    Returns:
        List of (q1, q6) solution pairs
    """
    # Extract target pose
    R06 = T06[:3, :3]
    P06 = T06[:3, 3]
    px, py, pz = P06
    
    r11, r12, r13 = R06[0, :]
    r21, r22, r23 = R06[1, :]
    r31, r32, r33 = R06[2, :]
    
    # Extract DH parameters
    alpha1 = dh_params['alpha1']
    d2 = dh_params['d2']
    d3 = dh_params['d3']
    d4 = dh_params['d4']
    alpha4 = dh_params['alpha4']
    alpha5 = dh_params['alpha5']
    
    ca1, sa1 = np.cos(alpha1), np.sin(alpha1)
    ca4 = np.cos(alpha4)
    ca5, sa5 = np.cos(alpha5), np.sin(alpha5)
    
    solutions_q1_q6 = []
    
    # Equation 2 (linear in q1):
    # -py*cos(q1)*sin(alpha1) + px*sin(q1)*sin(alpha1) + (-d2 - d3 - d4 + pz*cos(alpha1)) = 0
    # a*cos(q1) + b*sin(q1) + c = 0
    a_eq2 = -py * sa1
    b_eq2 = px * sa1
    c_eq2 = -d2 - d3 - d4 + pz * ca1
    
    if verbose:
        print(f"Equation 2 coefficients: a={a_eq2:.6f}, b={b_eq2:.6f}, c={c_eq2:.6f}")
    
    q1_solutions, is_arbitrary, _ = solve_trig_eq(a_eq2, b_eq2, c_eq2, verbose=verbose)
    
    if verbose:
        print(f"Found {len(q1_solutions)} solutions for q1: {np.degrees(q1_solutions)}")
    
    # For each q1 solution, solve equation 1 for q6
    for q1 in q1_solutions:
        c1, s1 = np.cos(q1), np.sin(q1)
        
        # Equation 1 (from Mathematica output):
        # -cos(alpha4) + sin(alpha1)*sin(q1)*(r13*cos(alpha5) + sin(alpha5)*(r12*cos(q6) + r11*sin(q6)))
        #               - cos(q1)*sin(alpha1)*(r23*cos(alpha5) + sin(alpha5)*(r22*cos(q6) + r21*sin(q6)))
        #               + cos(alpha1)*(r33*cos(alpha5) + sin(alpha5)*(r32*cos(q6) + r31*sin(q6))) = 0
        
        # Collect terms:
        # Constant term (no q6):
        const = -ca4 + sa1*s1*r13*ca5 - c1*sa1*r23*ca5 + ca1*r33*ca5
        
        # Coefficient of cos(q6):
        coeff_c6 = sa1*s1*r12*sa5 - c1*sa1*r22*sa5 + ca1*r32*sa5
        
        # Coefficient of sin(q6):
        coeff_s6 = sa1*s1*r11*sa5 - c1*sa1*r21*sa5 + ca1*r31*sa5
        
        if verbose:
            print(f"  For q1={np.degrees(q1):.2f}°:")
            print(f"    Equation 1 coefficients: a={coeff_c6:.6f}, b={coeff_s6:.6f}, c={const:.6f}")
        
        # Solve: coeff_c6*cos(q6) + coeff_s6*sin(q6) + const = 0
        q6_solutions, is_arb6, _ = solve_trig_eq(coeff_c6, coeff_s6, const, verbose=verbose)
        
        if verbose:
            print(f"    Found {len(q6_solutions)} solutions for q6: {np.degrees(q6_solutions)}")
        
        for q6 in q6_solutions:
            solutions_q1_q6.append((q1, q6))
    
    return solutions_q1_q6


def solve_q1_q6(T06, dh_params, verbose=False):
    """
    Solve for q1 and q6 from bilinear equations.
    
    Note: The decoupled solver (a5=d5=0) is less robust than the coupled solver.
    When a5≠0 or d5≠0, the general bilinear solver is used.
    
    For better robustness across edge cases, consider using non-zero a5 (e.g., a5=50mm).
    
    Returns:
        List of (q1, q6) solution pairs
    """
    # Check for special case: a5 = d5 = 0 (fully decoupled)
    # Only when BOTH a5 AND d5 are zero do the equations decouple
    # If only a5=0 but d5≠0, must use general bilinear solver
    a5 = dh_params['a5']
    d5 = dh_params['d5']
    a4 = dh_params['a4']
    
    if abs(a5) < 1e-10 and abs(d5) < 1e-10:
        if verbose:
            print("Special case: a5 = d5 = 0, equations decouple")
            print("Note: Decoupled case (a5=d5=0) has reduced robustness for some configurations")
        
        # Use decoupled solver
        solutions_q1_q6 = solve_q1_q6_decoupled(T06, dh_params, verbose=verbose)
        
        if verbose:
            print(f"Total {len(solutions_q1_q6)} (q1, q6) solution pairs from decoupled equations")
        
        return solutions_q1_q6
    
    # General case: solve bilinear system
    if verbose:
        print("General case: solving bilinear system")
    
    coeffs1, coeffs2 = compute_bilinear_coefficients(T06, dh_params)
    
    if verbose:
        print(f"\nEquation 1 coefficients:")
        print(f"  k0={coeffs1[0]:.6f}, kc1={coeffs1[1]:.6f}, ks1={coeffs1[2]:.6f}")
        print(f"  kc6={coeffs1[3]:.6f}, ks6={coeffs1[4]:.6f}")
        print(f"  kcc={coeffs1[5]:.6f}, kcs={coeffs1[6]:.6f}, ksc={coeffs1[7]:.6f}, kss={coeffs1[8]:.6f}")
        
        print(f"\nEquation 2 coefficients:")
        print(f"  k0={coeffs2[0]:.6f}, kc1={coeffs2[1]:.6f}, ks1={coeffs2[2]:.6f}")
        print(f"  kc6={coeffs2[3]:.6f}, ks6={coeffs2[4]:.6f}")
        print(f"  kcc={coeffs2[5]:.6f}, kcs={coeffs2[6]:.6f}, ksc={coeffs2[7]:.6f}, kss={coeffs2[8]:.6f}")
    
    # Use hybrid solver: resultant/companion matrix + two-line intersection
    # Create TwoEquationSystem from coefficient lists
    eq1 = EquationCoefficients(
        k0=coeffs1[0], k1c=coeffs1[1], k1s=coeffs1[2],
        k2c=coeffs1[3], k2s=coeffs1[4],
        kcc=coeffs1[5], kcs=coeffs1[6], ksc=coeffs1[7], kss=coeffs1[8]
    )
    eq2 = EquationCoefficients(
        k0=coeffs2[0], k1c=coeffs2[1], k1s=coeffs2[2],
        k2c=coeffs2[3], k2s=coeffs2[4],
        kcc=coeffs2[5], kcs=coeffs2[6], ksc=coeffs2[7], kss=coeffs2[8]
    )
    system = TwoEquationSystem(eq1=eq1, eq2=eq2)
    
    # Solve using hybrid method
    angle_solutions = solve_bilinear_sys(system, tol=1e-9)
    
    # Convert AngleSolution objects to (q1, q6) tuples
    # Skip solutions with free angles (shouldn't happen for well-posed IK)
    solutions_q1_q6 = []
    for sol in angle_solutions:
        if not sol.free_q1 and not sol.free_q2 and sol.q1 is not None and sol.q2 is not None:
            solutions_q1_q6.append((sol.q1, sol.q2))
    
    if verbose:
        print(f"\nFound {len(solutions_q1_q6)} (q1, q6) solutions from bilinear solver")
    
    # CRITICAL FIX: Tan-half substitution cannot represent ±180°
    # Explicitly test ±180° candidates for q1 and q6
    candidates_180 = [
        (np.pi, 0.0), (-np.pi, 0.0),      # q1 = ±180°, q6 = 0°
        (0.0, np.pi), (0.0, -np.pi),      # q1 = 0°, q6 = ±180°
        (np.pi, np.pi), (np.pi, -np.pi),  # q1 = ±180°, q6 = ±180°
        (-np.pi, np.pi), (-np.pi, -np.pi) # q1 = ±180°, q6 = ±180°
    ]
    
    tol = 1e-9
    for q1_test, q6_test in candidates_180:
        # Evaluate the bilinear equations
        c1, s1 = np.cos(q1_test), np.sin(q1_test)
        c6, s6 = np.cos(q6_test), np.sin(q6_test)
        
        res1 = (coeffs1[0] + coeffs1[1]*c1 + coeffs1[2]*s1 + coeffs1[3]*c6 + coeffs1[4]*s6 +
                coeffs1[5]*c1*c6 + coeffs1[6]*c1*s6 + coeffs1[7]*s1*c6 + coeffs1[8]*s1*s6)
        res2 = (coeffs2[0] + coeffs2[1]*c1 + coeffs2[2]*s1 + coeffs2[3]*c6 + coeffs2[4]*s6 +
                coeffs2[5]*c1*c6 + coeffs2[6]*c1*s6 + coeffs2[7]*s1*c6 + coeffs2[8]*s1*s6)
        
        if max(abs(res1), abs(res2)) < tol:
            # Normalize to [-π, π]
            q1_norm = np.arctan2(np.sin(q1_test), np.cos(q1_test))
            q6_norm = np.arctan2(np.sin(q6_test), np.cos(q6_test))
            
            # Check if this solution already exists
            is_duplicate = False
            for (q1_existing, q6_existing) in solutions_q1_q6:
                if abs(q1_norm - q1_existing) < 1e-6 and abs(q6_norm - q6_existing) < 1e-6:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                solutions_q1_q6.append((q1_norm, q6_norm))
                if verbose:
                    print(f"  Added ±180° solution: q1={np.rad2deg(q1_norm):.1f}°, q6={np.rad2deg(q6_norm):.1f}°")
    
    if verbose and len(solutions_q1_q6) > 0:
        print(f"\nTotal {len(solutions_q1_q6)} (q1, q6) solutions (including ±180° tests)")
    
    return solutions_q1_q6


def solve_q5(T06, dh_params, q1, q6, verbose=False):
    """
    Solve for q5 given q1 and q6.
    
    Uses T5Left = T5Right approach where:
    - T5Right = dhTrans[0, 0, 0, q2] . T23 . T34 (with 3rd column = [0,0,1,0])
    - T5Left = invDHTrans[α1, a1, d2, 0] . invT01 . T06 . invT56 . invT45
    
    From T5Left[[1,3]] and T5Left[[2,3]] = 0, solve for cos[q5] and sin[q5].
    """
    # Get DH parameters
    alpha1 = dh_params['alpha1']
    a1 = dh_params['a1']
    d2 = dh_params['d2']
    d1 = dh_params.get('d1', 0)
    alpha4 = dh_params['alpha4']
    a4 = dh_params['a4']
    d5 = dh_params['d5']
    alpha5 = dh_params['alpha5']
    a5 = dh_params['a5']
    d6 = dh_params.get('d6', 0)
    
    # Compute T5Left = invDHTrans[α1, a1, d2, 0] . invT01 . T06 . invT56 . invT45
    # where q5 is symbolic in invT45
    
    # invT01
    invT01 = inv_dh_transform(0, 0, d1, q1)
    
    # invDHTrans[α1, a1, d2, 0] - this moves the translation/rotation except q2 to LHS
    inv_T12_partial = inv_dh_transform(alpha1, a1, d2, 0)
    
    # invT56
    invT56 = inv_dh_transform(alpha5, a5, d6, q6)
    
    # For invT45, we need to express it with q5 as symbolic
    # invT45 has cos[q5] and sin[q5] terms
    # Instead, we'll extract the coefficients by computing T5Left symbolically
    
    # Compute the part that doesn't depend on q5
    T_temp = inv_T12_partial @ invT01 @ T06 @ invT56
    
    # Now T5Left = T_temp @ invT45(q5)
    # invT45 = inv(dhTrans[alpha4, a4, d5, q5])
    
    # The 3rd column of T5Left should be [0, 0, 1, 0]
    # T5Left[:, 2] = T_temp @ invT45_col3
    # where invT45_col3 is the 3rd column of invT45
    
    # invT45 3rd column is:
    ca4, sa4 = np.cos(alpha4), np.sin(alpha4)
    # [sin(q5)*sa4, cos(q5)*sa4, ca4, -d5*ca4]
    
    # So T5Left[i, 2] = sum_j T_temp[i,j] * invT45[j, 2]
    # invT45[:, 2] = [sin(q5)*sa4, cos(q5)*sa4, ca4, 0] (ignoring 4th row)
    
    # T5Left[0, 2] = T_temp[0,0]*sin(q5)*sa4 + T_temp[0,1]*cos(q5)*sa4 + T_temp[0,2]*ca4 = 0
    # T5Left[1, 2] = T_temp[1,0]*sin(q5)*sa4 + T_temp[1,1]*cos(q5)*sa4 + T_temp[1,2]*ca4 = 0
    # T5Left[2, 2] = T_temp[2,0]*sin(q5)*sa4 + T_temp[2,1]*cos(q5)*sa4 + T_temp[2,2]*ca4 = 1
    #
    # Note: T5Right[[3,3]] = 1 (not 0)!
    
    # Rearrange to A * [cos(q5), sin(q5)]^T = C (overdetermined 3x2 system)
    # The 3rd column of T5Right is [0, 0, 1, 0], so we need:
    # T5Left[[1,3]] = 0, T5Left[[2,3]] = 0, T5Left[[3,3]] = 1
    A = np.array([
        [T_temp[0, 1] * sa4, T_temp[0, 0] * sa4],
        [T_temp[1, 1] * sa4, T_temp[1, 0] * sa4],
        [T_temp[2, 1] * sa4, T_temp[2, 0] * sa4]
    ])
    C = np.array([
        -T_temp[0, 2] * ca4,
        -T_temp[1, 2] * ca4,
        1 - T_temp[2, 2] * ca4  # T5Left[[3,3]] = 1
    ])
    
    if verbose:
        print(f"\nSolving for q5 (overdetermined system):")
        print(f"  A (3x2) = \n{A}")
        print(f"  C = {C}")
    
    # Check for singularity: if A ≈ 0 and C ≈ 0, then q5 is arbitrary
    A_norm = np.linalg.norm(A, 'fro')
    C_norm = np.linalg.norm(C)
    
    if A_norm < 1e-8 and C_norm < 1e-6:
        # Singular configuration: q5 is arbitrary, set to 0
        if verbose:
            print(f"  ✓ Singular configuration detected (A_norm={A_norm:.3e}, C_norm={C_norm:.3e})")
            print(f"  ✓ q5 is arbitrary, setting q5 = 0°")
        q5_solutions = [0.0]
    else:
        # Use least squares to solve the overdetermined system
        # A * [cos(q5), sin(q5)]^T = C
        # Use pseudo-inverse for robustness
        try:
            # Compute pseudo-inverse solution
            trig_sol = np.linalg.lstsq(A, C, rcond=None)[0]
            cos_q5, sin_q5 = trig_sol[0], trig_sol[1]
            
            if verbose:
                print(f"  Least squares solution:")
                print(f"    cos(q5) = {cos_q5:.6f}")
                print(f"    sin(q5) = {sin_q5:.6f}")
            
            # Check trigonometric identity
            identity_error = cos_q5**2 + sin_q5**2 - 1.0
            
            if verbose:
                print(f"    cos²(q5) + sin²(q5) = {cos_q5**2 + sin_q5**2:.6f}")
                print(f"    Identity error: {identity_error:.6e}")
            
            # Verify residual
            residual = A @ np.array([cos_q5, sin_q5]) - C
            residual_norm = np.linalg.norm(residual)
            
            if verbose:
                print(f"    Residual norm: {residual_norm:.6e}")
            
            # If identity is satisfied reasonably well, use it
            if abs(identity_error) < 1e-3:
                q5 = np.arctan2(sin_q5, cos_q5)
                q5_solutions = [q5]
                
                if verbose:
                    print(f"  ✓ Valid solution: q5 = {np.degrees(q5):.2f}°")
            else:
                if verbose:
                    print(f"  ⚠ Trigonometric identity violated, normalizing...")
                
                # Normalize to satisfy identity
                norm = np.sqrt(cos_q5**2 + sin_q5**2)
                if norm > 1e-10:
                    cos_q5_norm = cos_q5 / norm
                    sin_q5_norm = sin_q5 / norm
                    q5 = np.arctan2(sin_q5_norm, cos_q5_norm)
                    q5_solutions = [q5]
                    
                    if verbose:
                        print(f"  ✓ Normalized solution: q5 = {np.degrees(q5):.2f}°")
                else:
                    if verbose:
                        print(f"  ✗ No valid solution found (norm too small)")
                    q5_solutions = []
            
        except np.linalg.LinAlgError:
            if verbose:
                print(f"  ✗ Failed to solve least squares problem")
            q5_solutions = []
    
    if verbose:
        print(f"  Found {len(q5_solutions)} q5 solutions: {[np.degrees(q) for q in q5_solutions]}°")
    
    return q5_solutions


def solve_planar_3r(T06, dh_params, q1, q6, q5, verbose=False):
    """
    Solve planar 3R problem for q2, q3, q4.
    
    Uses T5Left = T5Right approach where:
    - T5Right[[1,4]] = px = a2*cos[q2] + a3*cos[q2+q3]
    - T5Right[[2,4]] = py = a2*sin[q2] + a3*sin[q2+q3]
    - T5Right[[1,1]] = cos[q234]
    - T5Right[[2,1]] = sin[q234]
    
    Solve q3 from: px^2 + py^2 = a2^2 + a3^2 + 2*a2*a3*cos[q3]
    Then q2 from the position equations
    Then q4 = q234 - q2 - q3
    
    Args:
        T06: 4x4 target pose
        dh_params: Dictionary of DH parameters  
        q1, q6, q5: Already solved joint angles
        verbose: Print debug info
    
    Returns:
        List of (q2, q3, q4) solutions
    """
    # Get DH parameters
    alpha1 = dh_params['alpha1']
    a1 = dh_params['a1']
    d2 = dh_params['d2']
    d1 = dh_params.get('d1', 0)
    a2 = dh_params['a2']
    d3 = dh_params['d3']
    a3 = dh_params['a3']
    d4 = dh_params['d4']
    alpha4 = dh_params['alpha4']
    a4 = dh_params['a4']
    d5 = dh_params['d5']
    alpha5 = dh_params['alpha5']
    a5 = dh_params['a5']
    d6 = dh_params.get('d6', 0)
    
    # Compute T5Left = invDHTrans[α1, a1, d2, 0] . invT01 . T06 . invT56 . invT45
    invT01 = inv_dh_transform(0, 0, d1, q1)
    inv_T12_partial = inv_dh_transform(alpha1, a1, d2, 0)
    invT56 = inv_dh_transform(alpha5, a5, d6, q6)
    invT45 = inv_dh_transform(alpha4, a4, d5, q5)
    
    T5Left = inv_T12_partial @ invT01 @ T06 @ invT56 @ invT45
    
    # Extract px, py from T5Left position
    px = T5Left[0, 3]
    py = T5Left[1, 3]
    
    # Extract q234 from T5Left rotation
    r11 = T5Left[0, 0]
    r21 = T5Left[1, 0]
    q234 = np.arctan2(r21, r11)
    
    if verbose:
        print(f"\nSolving planar 3R:")
        print(f"  px = {px:.6f}, py = {py:.6f}")
        print(f"  q234 = {np.degrees(q234):.2f}°")
    
    solutions = []
    
    # Solve for q3 from: px^2 + py^2 = a2^2 + a3^2 + 2*a2*a3*cos[q3]
    # Note: This assumes both a2 and a3 are non-zero (checked in solve_ik_6r_3parallel)
    d_squared = px**2 + py**2
    cos_q3 = (d_squared - a2**2 - a3**2) / (2 * a2 * a3)
    
    if verbose:
        print(f"  d² = {d_squared:.6f}")
        print(f"  cos[q3] = {cos_q3:.6f}")
    
    # Check if solution exists - strict check with small numerical tolerance
    if abs(cos_q3) > 1.0 + 1e-6:
        if verbose:
            print(f"  No solution: |cos[q3]| = {abs(cos_q3):.6f} > 1 (unreachable)")
        return solutions
    
    # Clamp to [-1, 1] only for tiny numerical errors
    cos_q3 = np.clip(cos_q3, -1.0, 1.0)
    
    # Two solutions for q3: +/- arccos
    q3_candidates = [np.arccos(cos_q3), -np.arccos(cos_q3)]
    
    for q3 in q3_candidates:
        if verbose:
            print(f"\n  For q3 = {np.degrees(q3):.2f}°:")
        
        # Solve for q2 from position equations
        # px = a2*cos[q2] + a3*cos[q2+q3]
        # py = a2*sin[q2] + a3*sin[q2+q3]
        #
        # Expand: px = a2*cos[q2] + a3*(cos[q2]*cos[q3] - sin[q2]*sin[q3])
        #            = (a2 + a3*cos[q3])*cos[q2] - a3*sin[q3]*sin[q2]
        # Similarly: py = (a2 + a3*cos[q3])*sin[q2] + a3*sin[q3]*cos[q2]
        #
        # Matrix form: [[a2+a3*cos[q3], -a3*sin[q3]], [a3*sin[q3], a2+a3*cos[q3]]] * [cos[q2], sin[q2]]^T = [px, py]^T
        
        c3, s3 = np.cos(q3), np.sin(q3)
        A = np.array([
            [a2 + a3*c3, -a3*s3],
            [a3*s3, a2 + a3*c3]
        ])
        C = np.array([px, py])
        
        # Check if matrix is singular
        det_A = np.linalg.det(A)
        if abs(det_A) < 1e-10:
            if verbose:
                print(f"    Singular matrix, det(A) = {det_A:.2e}")
            continue
        
        # Solve for [cos[q2], sin[q2]]
        trig_sol = np.linalg.solve(A, C)
        cos_q2, sin_q2 = trig_sol[0], trig_sol[1]
        
        # Check trigonometric identity
        identity_error = cos_q2**2 + sin_q2**2 - 1.0
        if abs(identity_error) > 1e-4:
            if verbose:
                print(f"    Trig identity violated: cos²+sin²-1 = {identity_error:.2e}")
            continue
        
        q2 = np.arctan2(sin_q2, cos_q2)
        q4 = q234 - q2 - q3
        
        if verbose:
            print(f"    q2 = {np.degrees(q2):.2f}°")
            print(f"    q4 = {np.degrees(q4):.2f}°")
        
        solutions.append((q2, q3, q4))
    
    if verbose:
        print(f"  Found {len(solutions)} (q2, q3, q4) solutions")
    
    return solutions


def solve_ik_6r_3parallel(T06, dh_params, verbose=False):
    """
    Solve inverse kinematics for 6R robot with 3 parallel joints.
    
    Args:
        T06: 4x4 target pose
        dh_params: Dictionary of DH parameters
        verbose: Print debug information
    
    Returns:
        List of solution arrays [q1, q2, q3, q4, q5, q6]
    """
    # Pre-condition checks and preprocessing
    tol = 1e-9
    # Ensure joints 2,3,4 are parallel: alpha2 == alpha3 == 0
    alpha2 = dh_params.get('alpha2', 0.0)
    alpha3 = dh_params.get('alpha3', 0.0)
    if abs(alpha2) > tol or abs(alpha3) > tol:
        msg = (f"DH parameters do not match required parallel form: "
               f"alpha2={alpha2}, alpha3={alpha3} (expected 0).\n"
               "This solver only supports robots where joints 2,3,4 are parallel (alpha2=alpha3=0).")
        if verbose:
            print("WARNING:", msg)
        return []

    # The solver assumes alpha0=a0=d1=d6 == 0. If not, pre-process the target
    # pose T06 by removing the base and tool DH transforms:
    # T06' = inv_dh_transform(alpha0,a0,d1,0) . T06 . inv_dh_transform(0,0,d6,0)
    alpha0 = dh_params.get('alpha0', 0.0)
    a0 = dh_params.get('a0', 0.0)
    d1 = dh_params.get('d1', 0.0)
    d6 = dh_params.get('d6', 0.0)
    if abs(alpha0) > tol or abs(a0) > tol or abs(d1) > tol or abs(d6) > tol:
        if verbose:
            print("Preprocessing: removing base/tool offsets (alpha0,a0,d1,d6) before solving.")
        # Compute transformed target pose
        try:
            T06 = inv_dh_transform(alpha0, a0, d1, 0) @ T06 @ inv_dh_transform(0, 0, d6, 0)
        except Exception as e:
            if verbose:
                print(f"WARNING: failed to preprocess T06: {e}")
            return []
        # Use a modified DH parameter set with those entries zeroed so the solver
        # operates under its canonical assumptions
        dh_params = dh_params.copy()
        dh_params['alpha0'] = 0.0
        dh_params['a0'] = 0.0
        dh_params['d1'] = 0.0
        dh_params['d6'] = 0.0

    solutions = []
    
    # Step 1: Solve for q1 and q6
    q1_q6_solutions = solve_q1_q6(T06, dh_params, verbose=verbose)
    
    if len(q1_q6_solutions) == 0:
        if verbose:
            print("No solutions found for q1, q6")
        return solutions
    
    # Step 2: For each (q1, q6) pair, solve for q5, then q2, q3, q4
    for q1, q6 in q1_q6_solutions:
        if verbose:
            print(f"\nTrying q1={np.degrees(q1):.2f}°, q6={np.degrees(q6):.2f}°")
        
        # First solve for q5
        q5_solutions = solve_q5(T06, dh_params, q1, q6, verbose=verbose)
        
        if len(q5_solutions) == 0:
            if verbose:
                print(f"  No q5 solutions found for this (q1, q6) pair")
            continue
        
        # Then solve planar 3R for q2, q3, q4 (for each q5)
        for q5 in q5_solutions:
            q234_solutions = solve_planar_3r(T06, dh_params, q1, q6, q5, verbose=verbose)
            
            for q2, q3, q4 in q234_solutions:
                solution = np.array([q1, q2, q3, q4, q5, q6])
                solutions.append(solution)
    
    return solutions


def test_decoupled_case():
    """
    Test the IK solver with UR5e robot parameters (a5 = d5 = 0).
    This tests the decoupled equation solver.
    """
    print("=" * 80)
    print("TEST 1: DECOUPLED CASE (a5 = d5 = 0)")
    print("Testing with UR5e Robot Parameters")
    print("=" * 80)
    
    # UR5e DH parameters (in mm)
    dh_params = {
        'alpha1': np.pi/2,
        'a1': 0.0,
        'd2': 0.0,
        'a2': 425.0,
        'd3': 0.0,
        'a3': 392.25,
        'd4': 13.3,
        'alpha4': -np.pi/2,
        'a4': 0.0,
        'd5': 0.0,
        'alpha5': np.pi/2,
        'a5': 0.0
    }
    
    # Test joint angles
    q_test = np.array([30, -40, -45, 45, 25, 33]) * np.pi/180
    
    print(f"\nTest joint angles (degrees):")
    print(f"q = {np.degrees(q_test)}")
    print(f"\nDH Parameters: a5={dh_params['a5']}, d5={dh_params['d5']} (decoupled case)")
    
    # Compute forward kinematics
    T06_target = forward_kinematics(dh_params, q_test)
    
    print(f"\nTarget pose:")
    print(f"Position: {T06_target[:3, 3]}")
    print(f"Rotation matrix:")
    print(T06_target[:3, :3])
    
    # Solve IK
    print("\n" + "-" * 80)
    print("Solving IK...")
    print("-" * 80)
    
    solutions = solve_ik_6r_3parallel(T06_target, dh_params, verbose=True)
    
    print(f"\n" + "=" * 80)
    print(f"Found {len(solutions)} IK solution(s)")
    print("=" * 80)
    
    if len(solutions) > 0:
        for i, sol in enumerate(solutions):
            print(f"\nSolution {i+1}:")
            print(f"  q (degrees) = {np.degrees(sol)}")
            
            # Verify solution
            T06_sol = forward_kinematics(dh_params, sol)
            pos_error = np.linalg.norm(T06_sol[:3, 3] - T06_target[:3, 3])
            rot_error = np.linalg.norm(T06_sol[:3, :3] - T06_target[:3, :3], 'fro')
            
            print(f"  Position error: {pos_error:.6e}")
            print(f"  Rotation error: {rot_error:.6e}")
    
    return len(solutions) > 0


def test_coupled_case():
    """
    Test the IK solver with general robot parameters (a5 ≠ 0 or d5 ≠ 0).
    This tests the bilinear equation solver.
    """
    print("\n\n" + "=" * 80)
    print("TEST 2: COUPLED CASE (a5 ≠ 0, d5 ≠ 0)")
    print("Testing with Modified Robot Parameters")
    print("=" * 80)
    
    # Modified DH parameters with non-zero a5 and d5
    dh_params = {
        'alpha1': np.pi/2,
        'a1': 0.0,
        'd2': 0.0,
        'a2': 425.0,
        'd3': 0.0,
        'a3': 392.25,
        'd4': 13.3,
        'alpha4': -np.pi/2,
        'a4': 0.0,
        'd5': 99.7,  # Non-zero (like UR5e actual d6)
        'alpha5': np.pi/2,
        'a5': 0.0   # Keep zero for now
    }
    
    # Test joint angles
    q_test = np.array([30, -40, -45, 45, 25, 33]) * np.pi/180
    
    print(f"\nTest joint angles (degrees):")
    print(f"q = {np.degrees(q_test)}")
    print(f"\nDH Parameters: a5={dh_params['a5']}, d5={dh_params['d5']} (coupled case)")
    
    # Compute forward kinematics
    T06_target = forward_kinematics(dh_params, q_test)
    
    print(f"\nTarget pose:")
    print(f"Position: {T06_target[:3, 3]}")
    print(f"Rotation matrix:")
    print(T06_target[:3, :3])
    
    # Solve IK
    print("\n" + "-" * 80)
    print("Solving IK...")
    print("-" * 80)
    
    solutions = solve_ik_6r_3parallel(T06_target, dh_params, verbose=True)
    
    print(f"\n" + "=" * 80)
    print(f"Found {len(solutions)} IK solution(s)")
    print("=" * 80)
    
    if len(solutions) > 0:
        for i, sol in enumerate(solutions):
            print(f"\nSolution {i+1}:")
            print(f"  q (degrees) = {np.degrees(sol)}")
            
            # Verify solution
            T06_sol = forward_kinematics(dh_params, sol)
            pos_error = np.linalg.norm(T06_sol[:3, 3] - T06_target[:3, 3])
            rot_error = np.linalg.norm(T06_sol[:3, :3] - T06_target[:3, :3], 'fro')
            
            print(f"  Position error: {pos_error:.6e}")
            print(f"  Rotation error: {rot_error:.6e}")
    
    return len(solutions) > 0


def test_coupled_case_both_nonzero():
    """
    Test with both a5 and d5 non-zero.
    """
    print("\n\n" + "=" * 80)
    print("TEST 3: FULLY COUPLED CASE (a5 ≠ 0 AND d5 ≠ 0)")
    print("Testing with Both a5 and d5 Non-zero")
    print("=" * 80)
    
    # Modified DH parameters with both a5 and d5 non-zero
    dh_params = {
        'alpha1': np.pi/2,
        'a1': 0.0,
        'd2': 0.0,
        'a2': 425.0,
        'd3': 0.0,
        'a3': 392.25,
        'd4': 13.3,
        'alpha4': -np.pi/2,
        'a4': 0.0,
        'd5': 99.7,  # Non-zero
        'alpha5': np.pi/2,
        'a5': 95.0   # Non-zero
    }
    
    # Test joint angles
    q_test = np.array([30, -40, -45, 45, 25, 33]) * np.pi/180
    
    print(f"\nTest joint angles (degrees):")
    print(f"q = {np.degrees(q_test)}")
    print(f"\nDH Parameters: a5={dh_params['a5']}, d5={dh_params['d5']} (fully coupled case)")
    
    # Compute forward kinematics
    T06_target = forward_kinematics(dh_params, q_test)
    
    print(f"\nTarget pose:")
    print(f"Position: {T06_target[:3, 3]}")
    print(f"Rotation matrix:")
    print(T06_target[:3, :3])
    
    # Solve IK
    print("\n" + "-" * 80)
    print("Solving IK...")
    print("-" * 80)
    
    solutions = solve_ik_6r_3parallel(T06_target, dh_params, verbose=True)
    
    print(f"\n" + "=" * 80)
    print(f"Found {len(solutions)} IK solution(s)")
    print("=" * 80)
    
    if len(solutions) > 0:
        for i, sol in enumerate(solutions):
            print(f"\nSolution {i+1}:")
            print(f"  q (degrees) = {np.degrees(sol)}")
            
            # Verify solution
            T06_sol = forward_kinematics(dh_params, sol)
            pos_error = np.linalg.norm(T06_sol[:3, 3] - T06_target[:3, 3])
            rot_error = np.linalg.norm(T06_sol[:3, :3] - T06_target[:3, :3], 'fro')
            
            print(f"  Position error: {pos_error:.6e}")
            print(f"  Rotation error: {rot_error:.6e}")
    
    return len(solutions) > 0


if __name__ == "__main__":
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + " " * 20 + "6R 3-PARALLEL JOINT IK SOLVER TEST" + " " * 24 + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80 + "\n")
    
    # Run all tests
    test1_passed = test_decoupled_case()
    test2_passed = test_coupled_case()
    test3_passed = test_coupled_case_both_nonzero()
    
    # Summary
    print("\n\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Test 1 (Decoupled, a5=d5=0):        {'PASS' if test1_passed else 'FAIL'}")
    print(f"Test 2 (Coupled, d5≠0):             {'PASS' if test2_passed else 'FAIL'}")
    print(f"Test 3 (Fully coupled, a5≠0, d5≠0): {'PASS' if test3_passed else 'FAIL'}")
    print("=" * 80)
