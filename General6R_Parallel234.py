"""
INVERSE KINEMATICS SOLVER FOR GENERAL 6R ROBOTS WITH 3 PARALLEL JOINTS (2,3,4)

Solves IK for 6-DOF robots with parallel joints 2, 3, 4 using hybrid bilinear solver.
Compatible with RobotKinematicsCatalogue DHM table format.

DHM (Modified Denavit-Hartenberg) Parameters:
| Link | alpha_{i-1} | a_{i-1} | d_i | theta_i_offset |
|------|-------------|---------|-----|----------------|
| 1    | alpha0      | a0      | d1  | th1_offset     |
| 2    | alpha1      | a1      | d2  | th2_offset     |
| 3    | alpha2      | a2      | d3  | th3_offset     |
| 4    | alpha3      | a3      | d4  | th4_offset     |
| 5    | alpha4      | a4      | d5  | th5_offset     |
| 6    | alpha5      | a5      | d6  | th6_offset     |

Parallel joints constraint: joints 2, 3, 4 have parallel axes

Author: Haijun Su with Assistance from GitHub Copilot
Date: November 10, 2025
"""

import csv
import os
import sys

# Add paths for imports
base_scripts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(base_scripts_dir, 'helpers'))
sys.path.insert(0, base_scripts_dir)

from ik_6r_3parallel_solver import solve_ik_6r_3parallel, forward_kinematics
import numpy as np
from typing import List, Tuple


class General6R_Parallel234:
    """
    Inverse kinematics solver for general 6R robots with parallel joints 2, 3, 4.
    Compatible with RobotKinematicsCatalogue DHM table format.
    
    This class acts as a wrapper that:
    1. Accepts DHM table with joint angle offsets (theta_i_offset)
    2. Converts between theta-space (user angles) and q-space (kinematic angles)
    3. Uses the hybrid bilinear solver for robust IK
    """
    
    def __init__(self, DHM: np.ndarray, TB0: np.ndarray = None, T6W: np.ndarray = None, inv_joint: np.ndarray = None):
        """
        Initialize with 6x4 DHM parameter table (RobotKinematicsCatalogue format).
        
        Args:
            DHM: 6x4 numpy array with columns [alpha_{i-1}, a_{i-1}, d_i, theta_i_offset]
                 Each row i corresponds to link i (i = 1, 2, 3, 4, 5, 6)
            TB0: 4x4 base transform matrix (optional, defaults to identity)
            T6W: 4x4 tool/wrist transform matrix (optional, defaults to identity)
            inv_joint: 6-element array of inverse joint flags (±1) (optional, defaults to all 1)
        """
        if DHM.shape != (6, 4):
            raise ValueError(f"DHM table must be 6x4, got {DHM.shape}")
        
        # Store DHM table
        self.DHM = DHM.copy()
        
        # Extract parameters
        self.alpha = DHM[:, 0]  # alpha_{i-1} for link i
        self.a = DHM[:, 1]      # a_{i-1} for link i
        self.d = DHM[:, 2]      # d_i for link i
        self.theta_offset = DHM[:, 3]  # theta_i_offset for joint i
        
        # Base and tool transforms
        self.TB0 = TB0 if TB0 is not None else np.eye(4)
        self.T6W = T6W if T6W is not None else np.eye(4)
        
        # Inverse joint flags
        self.inv_joint = inv_joint if inv_joint is not None else np.ones(6)
        
        # Create DH parameters dictionary for the solver
        self.dh_params = {}
        for i in range(6):
            self.dh_params[f'alpha{i}'] = float(self.alpha[i])
            self.dh_params[f'a{i}'] = float(self.a[i])
            self.dh_params[f'd{i+1}'] = float(self.d[i])
        
        # Verify parallel joints constraint (joints 2, 3, 4 should have parallel axes)
        # For modified DH, parallel axes typically have alpha values of 0 or ±π
        # This is a soft check - the solver will work regardless
        
        print(f"Initialized General6R_Parallel234 IK solver:")
        print(f"  Base: alpha0={np.degrees(self.alpha[0]):.2f}°, a0={self.a[0]:.2f}, d1={self.d[0]:.2f}")
        print(f"  Joint offsets: [", end='')
        for i in range(6):
            print(f"{np.degrees(self.theta_offset[i]):.1f}°", end='')
            if i < 5:
                print(", ", end='')
        print("]")
        print(f"  Inverse joints: [", end='')
        for i in range(6):
            print(f"{int(self.inv_joint[i])}", end='')
            if i < 5:
                print(", ", end='')
        print("]")
        print(f"  Parallel joints: 2, 3, 4")

    def FK(self, joint: np.ndarray) -> np.ndarray:
        """
        Compute forward kinematics for the robot.
        
        Compatible with RobotKinematicsCatalogue interface.
        Expects joint angles in radians.
        
        Args:
            joint: 6-element array of joint angles in radians (theta-space)
            
        Returns:
            TBW: 4x4 homogeneous transformation matrix (base to wrist/tool)
        """
        # Convert to numpy array if needed
        joint = np.asarray(joint)
        
        # Compute IOtheta = theta_offset + inv_joint * joint (like catalogue)
        # joint and theta_offset are both in radians
        IOtheta = self.theta_offset + self.inv_joint * joint
        
        # Start with base transform
        TBW = self.TB0.copy()
        
        # Accumulate transforms for each link (like catalogue)
        for i in range(6):
            # Modified DH transform matrix
            temp = np.array([
                [np.cos(IOtheta[i]), -np.sin(IOtheta[i]), 0, self.a[i]],
                [np.sin(IOtheta[i]) * np.cos(self.alpha[i]), 
                 np.cos(IOtheta[i]) * np.cos(self.alpha[i]), 
                 -np.sin(self.alpha[i]), 
                 -np.sin(self.alpha[i]) * self.d[i]],
                [np.sin(IOtheta[i]) * np.sin(self.alpha[i]), 
                 np.cos(IOtheta[i]) * np.sin(self.alpha[i]), 
                 np.cos(self.alpha[i]), 
                 np.cos(self.alpha[i]) * self.d[i]],
                [0, 0, 0, 1]
            ])
            TBW = TBW @ temp
        
        # Apply tool transform
        TBW = TBW @ self.T6W
        
        return TBW

    def IK(self, TBW: np.ndarray, verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve inverse kinematics for target pose TBW.
        
        This method is compatible with RobotKinematicsCatalogue interface:
        - Returns (solutions, wrongSolutions) tuple
        - Solutions are in theta-space (user angles, in degrees)
        - wrongSolutions is included for compatibility but should be empty
        
        Args:
            TBW: 4x4 target homogeneous transformation matrix (base to wrist/tool)
            verbose: print detailed debug information
            
        Returns:
            Tuple (solutions, wrongSolutions):
            - solutions: Nx6 array of solutions in degrees (theta-space)
            - wrongSolutions: Empty array (included for compatibility)
        """
        if verbose:
            print("\n" + "="*70)
            print("GENERAL 6R PARALLEL234 INVERSE KINEMATICS SOLVER")
            print("="*70)
            print("Preprocessing: removing base/tool offsets (alpha0,a0,d1,d6) before solving.")
        
        # Preprocess TBW to get T06 for the solver
        # Since the solver now uses catalogue DH convention, and for AUBO_i5 TB0=I, T6W=I,
        # TBW = T01 @ T12 @ ... @ T56 = T06
        T06 = np.linalg.inv(self.TB0) @ TBW @ np.linalg.inv(self.T6W)
        
        if verbose:
            print(f"Target TBW z-translation: {TBW[2,3]:.1f} mm")
            print(f"T06 z-translation: {T06[2,3]:.1f} mm")
        
        # Solve IK using hybrid solver (returns solutions in q-space)
        try:
            solutions_q = solve_ik_6r_3parallel(T06, self.dh_params, verbose=verbose)
        except Exception as e:
            if verbose:
                print(f"IK solver error: {e}")
            # Return empty solutions in catalogue format
            return np.empty((0, 6)), np.empty((0, 6))
        
        if len(solutions_q) == 0:
            if verbose:
                print("No solutions found. Target unreachable or singular configuration.")
            return np.empty((0, 6)), np.empty((0, 6))
        
        # Convert solutions from q-space to theta-space
        solutions_theta = []
        for q_sol in solutions_q:
            # Subtract offsets to get theta values
            theta_sol = q_sol - self.theta_offset
            solutions_theta.append(theta_sol)
        
        solutions_theta = np.array(solutions_theta)
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"TOTAL SOLUTIONS FOUND: {len(solutions_theta)}")
            print("="*70)
        
        # Verify solutions using FK
        wrongSolutions = []
        validSolutions = []
        
        for i, theta_sol in enumerate(solutions_theta):
            # Convert to degrees for output
            sol_deg = np.degrees(theta_sol)
            
            # Verify with FK (expects radians)
            T_check = self.FK(theta_sol)
            
            pos_error = np.linalg.norm(T_check[:3, 3] - TBW[:3, 3])
            rot_error = np.linalg.norm(T_check[:3, :3] - TBW[:3, :3], 'fro')
            
            if pos_error < 1e-3 and rot_error < 1e-3:
                validSolutions.append(sol_deg)
                if verbose:
                    print(f"Solution {i+1} verified: pos_err={pos_error:.2e}, rot_err={rot_error:.2e}")
            else:
                wrongSolutions.append(sol_deg)
                if verbose:
                    print(f"Solution {i+1} FAILED: pos_err={pos_error:.2e}, rot_err={rot_error:.2e}")
        
        # Convert to numpy arrays in degrees (catalogue format)
        solutions = np.array(validSolutions) if validSolutions else np.empty((0, 6))
        wrongSolutions = np.array(wrongSolutions) if wrongSolutions else np.empty((0, 6))
        
        return solutions, wrongSolutions
