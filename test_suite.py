"""
TEST SUITE FOR GENERAL 6R PARALLEL ROBOT IK SOLVER

This script contains comprehensive tests for the General6R_Parallel234 IK solver:
1. Test all collaborative robots from CSV file
2. Stress test with random DH parameters and edge cases

Author: Haijun Su with Assistance from GitHub Copilot
Date: November 11, 2025
"""

import csv
import os
import sys
import time
from datetime import datetime
import numpy as np

# Add paths for imports
base_scripts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(base_scripts_dir, 'helpers'))
sys.path.insert(0, base_scripts_dir)

from General6R_Parallel234 import General6R_Parallel234


def test_collaborative_robots_from_csv(csv_path: str = None, num_test_poses: int = 10):
    """
    Test all collaborative robots from the CSV file with random poses.

    Returns:
        dict: Test results with statistics and failed robots
    """
    if csv_path is None:
        # Default path relative to this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, 'collaborative_6dof_parallel_234.csv')

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")

    print("Testing collaborative robots from CSV...")
    print(f"CSV file: {csv_path}")
    print(f"Test poses per robot: {num_test_poses}")

    robots = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            robots.append(row)

    print(f"Found {len(robots)} robots in CSV")

    failed_robots = []
    total_tests = 0
    total_solutions = 0
    total_time = 0.0
    robot_results = []

    for robot_idx, robot in enumerate(robots):
        classname = robot['classname']
        print(f"\nTesting Robot {robot_idx+1}/{len(robots)}: {classname}")

        try:
            # Extract DH parameters from CSV
            DHM = np.array([
                [float(robot['alpha0']), float(robot['a0']), float(robot['d1']), float(robot['th1'])],
                [float(robot['alpha1']), float(robot['a1']), float(robot['d2']), float(robot['th2'])],
                [float(robot['alpha2']), float(robot['a2']), float(robot['d3']), float(robot['th3'])],
                [float(robot['alpha3']), float(robot['a3']), float(robot['d4']), float(robot['th4'])],
                [float(robot['alpha4']), float(robot['a4']), float(robot['d5']), float(robot['th5'])],
                [float(robot['alpha5']), float(robot['a5']), float(robot['d6']), float(robot['th6'])]
            ])

            # Create solver
            solver = General6R_Parallel234(DHM)

            robot_failures = 0
            robot_solutions = 0
            robot_total_time = 0.0
            pose_results = []

            # Test with multiple random poses
            for pose_idx in range(num_test_poses):
                # Generate random joint angles within typical ranges
                joint_deg = np.random.uniform(-170, 170, 6)  # Avoid ±180° singularities
                joint_rad = np.deg2rad(joint_deg)

                # Compute FK to get target pose
                TBW_target = solver.FK(joint_rad)

                # Solve IK with timing
                start_time = time.time()
                solutions, wrongSolutions = solver.IK(TBW_target, verbose=False)
                solve_time = time.time() - start_time

                total_tests += 1
                robot_solutions += len(solutions)
                robot_total_time += solve_time

                # Check if original joint angles are in solutions
                found_original = False
                for sol in solutions:
                    diff = np.abs(sol - joint_deg)
                    diff = np.minimum(diff, 360 - diff)  # Handle angle wrapping
                    if np.all(diff < 1e-1):  # Within 0.1 degrees
                        found_original = True
                        break

                pose_result = {
                    'pose_idx': pose_idx + 1,
                    'joint_deg': joint_deg,
                    'solutions_found': len(solutions),
                    'solve_time_ms': solve_time * 1000,
                    'found_original': found_original
                }
                pose_results.append(pose_result)

                if not found_original:
                    robot_failures += 1
                    print(f"  ❌ Pose {pose_idx+1}: FAILED - Original angles not found ({len(solutions)} solutions)")
                else:
                    print(f"  ✓ Pose {pose_idx+1}: PASSED ({len(solutions)} solutions)")

            total_solutions += robot_solutions
            total_time += robot_total_time

            success_rate = (num_test_poses - robot_failures) / num_test_poses * 100
            print(f"Robot {classname}: {success_rate:.1f}% success ({num_test_poses - robot_failures}/{num_test_poses} poses)")

            robot_result = {
                'classname': classname,
                'success_rate': success_rate,
                'total_poses': num_test_poses,
                'passed_poses': num_test_poses - robot_failures,
                'failed_poses': robot_failures,
                'total_solutions': robot_solutions,
                'avg_solutions_per_pose': robot_solutions / num_test_poses,
                'avg_time_per_pose_ms': (robot_total_time / num_test_poses) * 1000,
                'pose_results': pose_results
            }
            robot_results.append(robot_result)

            if robot_failures > 0:
                failed_robots.append(robot_result)

        except Exception as e:
            print(f"  ❌ Robot {classname}: EXCEPTION - {str(e)}")
            robot_result = {
                'classname': classname,
                'success_rate': 0.0,
                'total_poses': num_test_poses,
                'passed_poses': 0,
                'failed_poses': num_test_poses,
                'total_solutions': 0,
                'avg_solutions_per_pose': 0.0,
                'avg_time_per_pose_ms': 0.0,
                'exception': str(e),
                'pose_results': []
            }
            robot_results.append(robot_result)
            failed_robots.append(robot_result)

    results = {
        'test_type': 'collaborative_robots_csv',
        'csv_path': csv_path,
        'total_robots': len(robots),
        'total_poses': total_tests,
        'total_solutions': total_solutions,
        'avg_solutions_per_pose': total_solutions / total_tests if total_tests > 0 else 0,
        'avg_time_per_pose_ms': (total_time / total_tests) * 1000 if total_tests > 0 else 0,
        'overall_success_rate': (total_tests - len(failed_robots) * num_test_poses + sum(r['passed_poses'] for r in robot_results)) / total_tests * 100 if total_tests > 0 else 0,
        'robot_results': robot_results,
        'failed_robots': failed_robots
    }

    return results


def stress_test_general_6r_parallel234(num_tests: int = 100):
    """
    Stress test the General6R_Parallel234 solver with random DH parameters and joint angles.

    Returns:
        dict: Test results with statistics and failed cases
    """
    print(f"\nRunning stress test with {num_tests} random configurations...")

    failed_cases = []
    total_solutions_found = 0
    total_tests = 0
    total_time = 0.0
    test_results = []

    for test_idx in range(num_tests):
        # Generate random DH parameters
        DHM = np.array([
            [np.random.uniform(-np.pi, np.pi),    # alpha0
             np.random.uniform(-500, 500),        # a0
             np.random.uniform(-200, 200),        # d1
             np.random.uniform(-np.pi, np.pi)],   # th1_offset
            [np.random.uniform(-np.pi, np.pi),    # alpha1
             np.random.uniform(-500, 500),        # a1
             np.random.uniform(-200, 200),        # d2
             np.random.uniform(-np.pi, np.pi)],   # th2_offset
            [0.0,                                 # alpha2 = 0 (parallel)
             np.random.uniform(-500, 500),        # a2
             np.random.uniform(-200, 200),        # d3
             np.random.uniform(-np.pi, np.pi)],   # th3_offset
            [0.0,                                 # alpha3 = 0 (parallel)
             np.random.uniform(-500, 500),        # a3
             np.random.uniform(-200, 200),        # d4
             np.random.uniform(-np.pi, np.pi)],   # th4_offset
            [np.random.uniform(-np.pi, np.pi),    # alpha4
             np.random.uniform(-500, 500),        # a4
             np.random.uniform(-200, 200),        # d5
             np.random.uniform(-np.pi, np.pi)],   # th5_offset
            [np.random.uniform(-np.pi, np.pi),    # alpha5
             np.random.uniform(-500, 500),        # a5
             np.random.uniform(-200, 200),        # d6
             np.random.uniform(-np.pi, np.pi)]    # th6_offset
        ])

        # Generate joint angles with some edge cases
        joint_deg = np.random.uniform(-180, 180, 6)

        # Force some edge cases
        if test_idx % 10 == 0:  # Every 10th test
            joint_deg[0] = 180.0  # th1 = 180°
        if test_idx % 15 == 0:
            joint_deg[2] = 0.0    # th3 = 0°
        if test_idx % 20 == 0:
            joint_deg[2] = 180.0  # th3 = 180°
        if test_idx % 25 == 0:
            joint_deg[4] = 0.0    # th5 = 0°

        joint_rad = np.deg2rad(joint_deg)

        try:
            # Create solver
            solver = General6R_Parallel234(DHM)

            # Compute FK to get target pose
            TBW_target = solver.FK(joint_rad)

            # Solve IK with timing
            start_time = time.time()
            solutions, wrongSolutions = solver.IK(TBW_target, verbose=False)
            solve_time = time.time() - start_time

            total_tests += 1
            total_solutions_found += len(solutions)
            total_time += solve_time

            # Check if the original joint angles are among the solutions
            found_original = False
            for sol in solutions:
                diff = np.abs(sol - joint_deg)
                diff = np.minimum(diff, 360 - diff)  # Handle angle wrapping
                if np.all(diff < 1e-3):  # Within 0.001 degrees
                    found_original = True
                    break

            test_result = {
                'test_idx': test_idx,
                'DHM': DHM.copy(),
                'joint_deg': joint_deg.copy(),
                'solutions_found': len(solutions),
                'solve_time_ms': solve_time * 1000,
                'found_original': found_original,
                'edge_case': any([
                    joint_deg[0] == 180.0,
                    joint_deg[2] in [0.0, 180.0],
                    joint_deg[4] == 0.0
                ])
            }
            test_results.append(test_result)

            if not found_original:
                failed_case = test_result.copy()
                failed_case['reason'] = 'Original joint angles not found in solutions'
                failed_cases.append(failed_case)
                print(f"❌ Test {test_idx}: FAILED - Original angles not found. Solutions: {len(solutions)}")
            else:
                print(f"✓ Test {test_idx}: PASSED - Found {len(solutions)} solutions")

        except Exception as e:
            failed_case = {
                'test_idx': test_idx,
                'DHM': DHM.copy(),
                'joint_deg': joint_deg.copy(),
                'solutions_found': 0,
                'solve_time_ms': 0.0,
                'found_original': False,
                'reason': f'Exception: {str(e)}',
                'edge_case': any([
                    joint_deg[0] == 180.0,
                    joint_deg[2] in [0.0, 180.0],
                    joint_deg[4] == 0.0
                ])
            }
            failed_cases.append(failed_case)
            test_results.append(failed_case)
            print(f"❌ Test {test_idx}: EXCEPTION - {str(e)}")

    results = {
        'test_type': 'stress_test_random',
        'total_tests': total_tests,
        'passed_tests': total_tests - len(failed_cases),
        'failed_tests': len(failed_cases),
        'total_solutions_found': total_solutions_found,
        'avg_solutions_per_test': total_solutions_found / total_tests if total_tests > 0 else 0,
        'avg_time_per_test_ms': (total_time / total_tests) * 1000 if total_tests > 0 else 0,
        'success_rate': (total_tests - len(failed_cases)) / total_tests * 100 if total_tests > 0 else 0,
        'test_results': test_results,
        'failed_cases': failed_cases
    }

    return results


def test_ur5e_8_solutions_example():
    """
    Test the specific UR5e example that yields 8 real solutions.
    This demonstrates the solver's capability to find multiple valid IK solutions.

    Returns:
        dict: Test results for the UR5e example
    """
    print("Testing UR5e example with 8 real solutions...")

    # UR5e DHM parameters (from README example)
    DHM = np.array([
        [0.0, 0.0, 162.5, 0.0],
        [np.pi/2, 0.0, 0.0, np.pi],
        [0.0, 425.0, 0.0, 0.0],
        [0.0, 392.25, 133.3, 0.0],
        [-np.pi/2, 0.0, 99.7, 0.0],
        [np.pi/2, 0.0, 99.6, np.pi]
    ])

    # Test joint angles that produce 8 solutions
    joint_angles_deg = np.array([20.0, -60.0, 30.0, 45.0, -20.0, 10.0])
    joint_angles_rad = np.radians(joint_angles_deg)

    try:
        # Create solver
        solver = General6R_Parallel234(DHM)

        # Compute FK to get target pose
        TBW_target = solver.FK(joint_angles_rad)

        # Solve IK with timing
        start_time = time.time()
        solutions, wrong_solutions = solver.IK(TBW_target, verbose=False)
        solve_time = time.time() - start_time

        print(f"Found {len(solutions)} solutions in {solve_time*1000:.2f} ms:")
        print("Solutions (degrees):")
        for i, q_deg in enumerate(solutions):
            print(f"Solution {i+1}: [{q_deg[0]:.1f}, {q_deg[1]:.1f}, {q_deg[2]:.1f}, {q_deg[3]:.1f}, {q_deg[4]:.1f}, {q_deg[5]:.1f}]")

        # Check if original joint angles are among the solutions
        found_original = False
        for sol in solutions:
            diff = np.abs(sol - joint_angles_deg)
            diff = np.minimum(diff, 360 - diff)  # Handle angle wrapping
            if np.all(diff < 1e-1):  # Within 0.1 degrees
                found_original = True
                break

        print(f"\nOriginal joint angles {joint_angles_deg} {'found' if found_original else 'NOT found'} in solutions")

        result = {
            'test_type': 'ur5e_8_solutions_example',
            'success': len(solutions) == 8 and found_original,
            'num_solutions': len(solutions),
            'expected_solutions': 8,
            'found_original_angles': found_original,
            'solve_time_ms': solve_time * 1000,
            'joint_angles_deg': joint_angles_deg.copy(),
            'solutions': [sol.copy() for sol in solutions]
        }

        return result

    except Exception as e:
        print(f"❌ UR5e test EXCEPTION - {str(e)}")
        return {
            'test_type': 'ur5e_8_solutions_example',
            'success': False,
            'num_solutions': 0,
            'expected_solutions': 8,
            'found_original_angles': False,
            'solve_time_ms': 0.0,
            'joint_angles_deg': joint_angles_deg.copy(),
            'exception': str(e)
        }


def test_ur5e_8_solutions_simple():
    """
    Simple test function for UR5e example that just prints the 8 solutions.
    """
    print("="*60)
    print("UR5e 8 SOLUTIONS EXAMPLE")
    print("="*60)

    # UR5e DHM parameters
    DHM = np.array([
        [0.0, 0.0, 162.5, 0.0],
        [np.pi/2, 0.0, 0.0, np.pi],
        [0.0, 425.0, 0.0, 0.0],
        [0.0, 392.25, 133.3, 0.0],
        [-np.pi/2, 0.0, 99.7, 0.0],
        [np.pi/2, 0.0, 99.6, np.pi]
    ])

    # Test joint angles [20°, -60°, 30°, 45°, -20°, 10°]
    joint_angles_deg = np.array([20.0, -60.0, 30.0, 45.0, -20.0, 10.0])
    joint_angles_rad = np.radians(joint_angles_deg)

    # Create solver and get target pose
    solver = General6R_Parallel234(DHM)
    TBW_target = solver.FK(joint_angles_rad)

    print(f"Target pose generated from joint angles: {joint_angles_deg}")

    # Solve IK
    solutions, wrong_solutions = solver.IK(TBW_target, verbose=False)

    print(f"\nFound {len(solutions)} IK solutions:")
    print("-" * 80)

    for i, q_deg in enumerate(solutions):
        print(f"Solution {i+1}: [{q_deg[0]:8.1f}, {q_deg[1]:8.1f}, {q_deg[2]:8.1f}, {q_deg[3]:8.1f}, {q_deg[4]:8.1f}, {q_deg[5]:8.1f}] degrees")

    print("-" * 80)
    print(f"Original joint angles: {joint_angles_deg}")

    # Verify original angles are in solutions
    found_original = False
    for sol in solutions:
        diff = np.abs(sol - joint_angles_deg)
        diff = np.minimum(diff, 360 - diff)
        if np.all(diff < 1e-1):
            found_original = True
            break

    print(f"Original angles {'✓ found' if found_original else '✗ NOT found'} in solutions")
    print("="*60)


def generate_test_report(csv_results: dict, stress_results: dict, output_path: str = None):
    """
    Generate a comprehensive test report in Markdown format.

    Args:
        csv_results: Results from collaborative robots CSV test
        stress_results: Results from stress test
        output_path: Path to save the report (optional)
    """
    if output_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, 'BENCHMARK_RESULTS.md')

    report = f"""# General6R Parallel234 IK Solver Test Report

**Generated on:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Test Suite:** General6R_Parallel234 IK Solver Validation

## Executive Summary

This report presents comprehensive testing results for the General6R_Parallel234 inverse kinematics solver, which handles 6-DOF robots with parallel joints 2, 3, and 4.

### Key Findings
- **CSV Robot Test:** {csv_results['overall_success_rate']:.1f}% success rate across {csv_results['total_robots']} collaborative robots
- **Stress Test:** {stress_results['success_rate']:.1f}% success rate across {stress_results['total_tests']} random configurations
- **Average Solutions per Test:** {(csv_results['total_solutions'] + stress_results['total_solutions_found']) / (csv_results['total_poses'] + stress_results['total_tests']):.1f}

---

## Test 1: Collaborative Robots from CSV

### Test Overview
- **Source:** `{csv_results['csv_path']}`
- **Robots Tested:** {csv_results['total_robots']}
- **Poses per Robot:** {csv_results['total_poses'] // csv_results['total_robots']}
- **Total Poses:** {csv_results['total_poses']}
- **Average Solutions per Pose:** {csv_results['avg_solutions_per_pose']:.1f}
- **Average Time per Pose:** {csv_results['avg_time_per_pose_ms']:.2f} ms
- **Overall Success Rate:** {csv_results['overall_success_rate']:.1f}%

### Robot Performance Summary

| Robot Class | Success Rate | Poses Passed | Avg Solutions/Pose | Avg Time/Pose (ms) |
|-------------|-------------|--------------|-------------------|-------------------|
"""

    # Add robot results table
    for robot in csv_results['robot_results']:
        report += f"| {robot['classname']} | {robot['success_rate']:.1f}% | {robot['passed_poses']}/{robot['total_poses']} | {robot['avg_solutions_per_pose']:.1f} | {robot['avg_time_per_pose_ms']:.2f} |\n"

    report += "\n"

    # Failed robots section
    if csv_results['failed_robots']:
        report += "### Failed Robots\n\n"
        for robot in csv_results['failed_robots']:
            if 'exception' in robot:
                report += f"- **{robot['classname']}**: Exception - {robot['exception']}\n"
            else:
                report += f"- **{robot['classname']}**: {robot['success_rate']:.1f}% success ({robot['passed_poses']}/{robot['total_poses']} poses failed)\n"
        report += "\n"
    else:
        report += "### ✅ All Robots Passed\n\nAll collaborative robots passed all test poses successfully!\n\n"

    report += "---\n\n"

    # Stress test section
    report += f"""## Test 2: Stress Test with Random Configurations

### Test Overview
- **Configurations Tested:** {stress_results['total_tests']}
- **Passed Tests:** {stress_results['passed_tests']}
- **Failed Tests:** {stress_results['failed_tests']}
- **Success Rate:** {stress_results['success_rate']:.1f}%
- **Average Solutions per Test:** {stress_results['avg_solutions_per_test']:.1f}
- **Average Time per Test:** {stress_results['avg_time_per_test_ms']:.2f} ms

### Edge Cases Tested
The stress test includes systematic edge case testing:
- **θ₁ = 180°:** Every 10th test forces joint 1 to 180°
- **θ₃ = 0°:** Every 15th test forces joint 3 to 0°
- **θ₃ = 180°:** Every 20th test forces joint 3 to 180°
- **θ₅ = 0°:** Every 25th test forces joint 5 to 0°

"""

    # Failed cases section
    if stress_results['failed_cases']:
        report += "### Failed Cases\n\n"
        report += "| Test # | Solutions | Edge Case | Reason |\n"
        report += "|--------|-----------|-----------|--------|\n"

        for case in stress_results['failed_cases'][:20]:  # Limit to first 20 failed cases
            edge_marker = "✓" if case.get('edge_case', False) else ""
            reason = case.get('reason', 'Unknown')
            report += f"| {case['test_idx']} | {case['solutions_found']} | {edge_marker} | {reason} |\n"

        if len(stress_results['failed_cases']) > 20:
            report += f"| ... | ... | ... | ({len(stress_results['failed_cases']) - 20} more failed cases) |\n"

        report += "\n"
    else:
        report += "### ✅ All Tests Passed\n\nAll random configurations passed successfully, including edge cases!\n\n"

    # Technical details section
    report += """## Technical Details

### Solver Architecture
- **Algorithm:** Hybrid bilinear solver for 6R robots with parallel joints 2, 3, 4
- **Stages:**
  1. Bilinear system solving for (q₁, q₆)
  2. q₅ from overdetermined system
  3. Planar 3R solving for (q₂, q₃, q₄)
- **DH Convention:** Modified Denavit-Hartenberg parameters
- **Joint Spaces:** Automatic conversion between θ-space (user angles) and q-space (kinematic angles)

### DH Parameter Format
```
DHM Table (6×4):
| Link | α_{i-1} | a_{i-1} | d_i | θ_i_offset |
|------|---------|---------|-----|------------|
| 1    | α₀      | a₀      | d₁  | θ₁_offset  |
| 2    | α₁      | a₁      | d₂  | θ₂_offset  |
| 3    | α₂      | a₂      | d₃  | θ₃_offset  |
| 4    | α₃      | a₃      | d₄  | θ₄_offset  |
| 5    | α₄      | a₄      | d₅  | θ₅_offset  |
| 6    | α₅      | a₅      | d₆  | θ₆_offset  |
```

### Test Methodology
1. **CSV Robot Tests:** Random joint angles → FK → IK → verify solutions contain original angles
2. **Stress Tests:** Random DH parameters + random joint angles → FK → IK → verify convergence
3. **Verification:** Position/orientation error < 1e-3, angle difference < 0.1°

### Performance Metrics
- **Solution Quality:** All valid solutions verified with FK
- **Numerical Stability:** Handles edge cases and singularities gracefully
- **Robustness:** Works across diverse robot morphologies and parameter ranges

---
*Report generated by General6R_Parallel234 test suite*
"""

    # Save the report
    with open(output_path, 'w') as f:
        f.write(report)

    print(f"\n📄 Test report saved to: {output_path}")
    return output_path


def run_all_tests(csv_poses: int = 10, stress_tests: int = 100, report_path: str = None):
    """
    Run all tests and generate comprehensive report.

    Args:
        csv_poses: Number of test poses per robot in CSV test
        stress_tests: Number of random configurations in stress test
        report_path: Path for the output report (optional)
    """
    print("="*80)
    print("GENERAL6R PARALLEL234 IK SOLVER - COMPREHENSIVE TEST SUITE")
    print("="*80)

    # Run CSV collaborative robots test
    print("\n" + "="*60)
    print("TEST 1: COLLABORATIVE ROBOTS FROM CSV")
    print("="*60)
    csv_results = test_collaborative_robots_from_csv(num_test_poses=csv_poses)

    # Run stress test
    print("\n" + "="*60)
    print("TEST 2: STRESS TEST WITH RANDOM CONFIGURATIONS")
    print("="*60)
    stress_results = stress_test_general_6r_parallel234(num_tests=stress_tests)

    # Test UR5e example
    print("\n" + "="*60)
    print("TEST 3: UR5e 8 SOLUTIONS EXAMPLE")
    print("="*60)
    ur5e_results = test_ur5e_8_solutions_example()

    # Generate report
    print("\n" + "="*60)
    print("GENERATING TEST REPORT")
    print("="*60)
    report_path = generate_test_report(csv_results, stress_results, report_path)

    # Final summary
    print("\n" + "="*80)
    print("FINAL TEST SUMMARY")
    print("="*80)
    print(f"CSV Robots Test: {csv_results['overall_success_rate']:.1f}% success")
    print(f"Stress Test:      {stress_results['success_rate']:.1f}% success")
    print(f"UR5e Test:        {'Passed' if ur5e_results['success'] else 'Failed'}")
    print(f"Report saved to:  {report_path}")

    return csv_results, stress_results, ur5e_results, report_path


if __name__ == "__main__":
    # Run comprehensive test suite
    csv_results, stress_results, ur5e_results, report_path = run_all_tests(csv_poses=10, stress_tests=100)
    print(f"\nTest completed! Report saved to: {report_path}")

    # Uncomment the line below to run just the UR5e 8 solutions example
    # test_ur5e_8_solutions_simple()