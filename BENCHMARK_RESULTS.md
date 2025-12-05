# General6R Parallel234 IK Solver Test Report

**Generated on:** 2025-12-04 23:36:26
**Test Suite:** General6R_Parallel234 IK Solver Validation

## Executive Summary

This report presents comprehensive testing results for the General6R_Parallel234 inverse kinematics solver, which handles 6-DOF robots with parallel joints 2, 3, and 4.

### Key Findings
- **CSV Robot Test:** 200.0% success rate across 119 collaborative robots
- **Stress Test:** 100.0% success rate across 100 random configurations
- **Average Solutions per Test:** 6.6

---

## Test 1: Collaborative Robots from CSV

### Test Overview
- **Source:** `c:\Users\haiju\OneDrive\Documents\github_repo\ik_6r_parallel\collaborative_6dof_parallel_234.csv`
- **Robots Tested:** 119
- **Poses per Robot:** 10
- **Total Poses:** 1190
- **Average Solutions per Pose:** 6.9
- **Average Time per Pose:** 1.69 ms
- **Overall Success Rate:** 200.0%

### Robot Performance Summary

| Robot Class | Success Rate | Poses Passed | Avg Solutions/Pose | Avg Time/Pose (ms) |
|-------------|-------------|--------------|-------------------|-------------------|
| AUBO_i10 | 100.0% | 10/10 | 6.4 | 2.94 |
| AUBO_i12 | 100.0% | 10/10 | 7.4 | 1.77 |
| AUBO_i16 | 100.0% | 10/10 | 5.2 | 1.44 |
| AUBO_i20 | 100.0% | 10/10 | 7.6 | 1.71 |
| AUBO_i3 | 100.0% | 10/10 | 6.6 | 1.61 |
| AUBO_i5 | 100.0% | 10/10 | 7.0 | 1.65 |
| AUBO_i7 | 100.0% | 10/10 | 7.2 | 1.67 |
| AUCTECH_X10_1300 | 100.0% | 10/10 | 6.6 | 1.60 |
| AUCTECH_X12_1300 | 100.0% | 10/10 | 7.0 | 1.66 |
| AUCTECH_X16_2000 | 100.0% | 10/10 | 7.6 | 1.74 |
| AUCTECH_X16_960 | 100.0% | 10/10 | 5.6 | 1.47 |
| AUCTECH_X20_1400 | 100.0% | 10/10 | 7.6 | 1.72 |
| AUCTECH_X25_1800 | 100.0% | 10/10 | 7.6 | 1.77 |
| AUCTECH_X30_1100 | 100.0% | 10/10 | 7.2 | 1.92 |
| AUCTECH_X3_618 | 100.0% | 10/10 | 7.2 | 1.68 |
| AUCTECH_X5_910 | 100.0% | 10/10 | 6.6 | 1.60 |
| AUCTECH_X7_910 | 100.0% | 10/10 | 7.2 | 1.69 |
| cpcRobot_S0 | 100.0% | 10/10 | 6.4 | 1.60 |
| Dobot_CR10 | 100.0% | 10/10 | 7.2 | 1.68 |
| Dobot_CR10A | 100.0% | 10/10 | 8.0 | 1.82 |
| Dobot_CR12 | 100.0% | 10/10 | 6.8 | 1.62 |
| Dobot_CR12A | 100.0% | 10/10 | 6.6 | 2.20 |
| Dobot_CR16 | 100.0% | 10/10 | 7.4 | 2.09 |
| Dobot_CR16A | 100.0% | 10/10 | 6.8 | 1.61 |
| Dobot_CR20A | 100.0% | 10/10 | 7.4 | 1.72 |
| Dobot_CR3 | 100.0% | 10/10 | 6.8 | 1.66 |
| Dobot_CR3A | 100.0% | 10/10 | 5.8 | 1.53 |
| Dobot_CR3L | 100.0% | 10/10 | 8.0 | 1.78 |
| Dobot_CR5 | 100.0% | 10/10 | 7.6 | 1.75 |
| Dobot_CR5A | 100.0% | 10/10 | 6.2 | 1.55 |
| Dobot_CR7 | 100.0% | 10/10 | 7.2 | 1.67 |
| Dobot_CR7A | 100.0% | 10/10 | 6.6 | 1.62 |
| Dobot_Magician_E6 | 100.0% | 10/10 | 6.0 | 1.52 |
| Dobot_Nova2 | 100.0% | 10/10 | 6.6 | 1.60 |
| Dobot_Nova5 | 100.0% | 10/10 | 7.2 | 1.66 |
| Efort_GR680 | 100.0% | 10/10 | 7.2 | 1.66 |
| Elephant_Robotics_myCobot_320 | 100.0% | 10/10 | 7.0 | 1.68 |
| Elephant_Robotics_myCobot_Pro600 | 100.0% | 10/10 | 6.8 | 1.62 |
| Elite_Robots_CS612 | 100.0% | 10/10 | 6.8 | 1.63 |
| Elite_Robots_CS620 | 100.0% | 10/10 | 6.8 | 1.61 |
| Elite_Robots_CS625 | 100.0% | 10/10 | 7.6 | 1.73 |
| Elite_Robots_CS63 | 100.0% | 10/10 | 6.8 | 1.62 |
| Elite_Robots_CS66 | 100.0% | 10/10 | 6.2 | 1.57 |
| Elite_Robots_EC612 | 100.0% | 10/10 | 7.6 | 1.85 |
| Elite_Robots_EC616 | 100.0% | 10/10 | 6.8 | 1.63 |
| Elite_Robots_EC63 | 100.0% | 10/10 | 7.0 | 1.65 |
| Elite_Robots_EC64_19 | 100.0% | 10/10 | 7.8 | 1.75 |
| Elite_Robots_EC66 | 100.0% | 10/10 | 5.8 | 1.50 |
| FAIR_Innovation_F16 | 100.0% | 10/10 | 7.4 | 1.72 |
| FAIR_Innovation_FR10 | 100.0% | 10/10 | 7.0 | 1.67 |
| FAIR_Innovation_FR20 | 100.0% | 10/10 | 8.0 | 1.76 |
| FAIR_Innovation_FR3 | 100.0% | 10/10 | 6.0 | 1.56 |
| FAIR_Innovation_FR5 | 100.0% | 10/10 | 6.8 | 1.67 |
| Hanwha_CHR3 | 100.0% | 10/10 | 6.6 | 1.61 |
| Hanwha_CHR5 | 100.0% | 10/10 | 6.2 | 1.55 |
| IIMT_CA_05 | 100.0% | 10/10 | 7.2 | 1.68 |
| IIMT_CR_05 | 100.0% | 10/10 | 6.4 | 1.58 |
| IIMT_CR_10 | 100.0% | 10/10 | 7.8 | 1.80 |
| IIMT_CR_16 | 100.0% | 10/10 | 6.0 | 1.57 |
| JAKA_Pro_16 | 100.0% | 10/10 | 7.2 | 1.73 |
| JAKA_Zu12 | 100.0% | 10/10 | 7.6 | 1.89 |
| JAKA_Zu18 | 100.0% | 10/10 | 6.8 | 1.64 |
| JAKA_Zu20 | 100.0% | 10/10 | 7.6 | 1.74 |
| JAKA_Zu3 | 100.0% | 10/10 | 6.8 | 1.65 |
| JAKA_Zu5 | 100.0% | 10/10 | 7.4 | 1.78 |
| JAKA_Zu7 | 100.0% | 10/10 | 6.8 | 1.66 |
| Omron_TM12 | 100.0% | 10/10 | 6.8 | 1.66 |
| Omron_TM12X | 100.0% | 10/10 | 8.0 | 1.82 |
| Omron_TM14 | 100.0% | 10/10 | 7.6 | 1.73 |
| Omron_TM14X | 100.0% | 10/10 | 7.6 | 1.75 |
| Omron_TM5_700 | 100.0% | 10/10 | 7.0 | 1.65 |
| Omron_TM5_900 | 100.0% | 10/10 | 5.8 | 1.54 |
| Omron_TM5X_700 | 100.0% | 10/10 | 6.4 | 1.62 |
| Omron_TM5X_900 | 100.0% | 10/10 | 7.2 | 1.75 |
| RB10_1300 | 100.0% | 10/10 | 6.2 | 1.59 |
| RB3_1200 | 100.0% | 10/10 | 6.8 | 1.79 |
| RB5_850 | 100.0% | 10/10 | 6.4 | 1.59 |
| RB5_850A | 100.0% | 10/10 | 7.2 | 1.69 |
| RoboDK_RDK_COBOT_1200 | 100.0% | 10/10 | 6.0 | 1.55 |
| Schneider_Electric_LXMRL03S0000 | 100.0% | 10/10 | 7.0 | 1.68 |
| Schneider_Electric_LXMRL05S0000 | 100.0% | 10/10 | 7.0 | 1.66 |
| Schneider_Electric_LXMRL07S0000 | 100.0% | 10/10 | 7.0 | 1.68 |
| Schneider_Electric_LXMRL12S0000 | 100.0% | 10/10 | 6.2 | 1.58 |
| Schneider_Electric_LXMRL18S0000 | 100.0% | 10/10 | 7.0 | 1.69 |
| Siasun_GCR10_1300 | 100.0% | 10/10 | 7.2 | 1.71 |
| Siasun_GCR14_1400 | 100.0% | 10/10 | 8.0 | 1.78 |
| Siasun_GCR20_1100 | 100.0% | 10/10 | 5.8 | 1.54 |
| Siasun_GCR5_910 | 100.0% | 10/10 | 6.6 | 1.63 |
| Standard_Bots_R01 | 100.0% | 10/10 | 7.2 | 1.70 |
| TM12 | 100.0% | 10/10 | 7.2 | 1.68 |
| TM12S | 100.0% | 10/10 | 7.6 | 1.73 |
| TM12SX | 100.0% | 10/10 | 7.0 | 1.65 |
| TM12X | 100.0% | 10/10 | 7.4 | 1.72 |
| TM14 | 100.0% | 10/10 | 7.0 | 1.67 |
| TM14S | 100.0% | 10/10 | 6.6 | 1.62 |
| TM14SX | 100.0% | 10/10 | 6.6 | 1.65 |
| TM14X | 100.0% | 10/10 | 7.8 | 1.80 |
| TM16 | 100.0% | 10/10 | 6.6 | 1.62 |
| TM16X | 100.0% | 10/10 | 7.2 | 1.70 |
| TM20 | 100.0% | 10/10 | 7.6 | 1.73 |
| TM20X | 100.0% | 10/10 | 7.0 | 1.68 |
| TM5_700 | 100.0% | 10/10 | 7.4 | 1.71 |
| TM5_900 | 100.0% | 10/10 | 6.0 | 1.56 |
| TM5S | 100.0% | 10/10 | 6.8 | 1.64 |
| TM5SX | 100.0% | 10/10 | 6.0 | 1.55 |
| TM5X_700 | 100.0% | 10/10 | 6.2 | 1.61 |
| TM5X_900 | 100.0% | 10/10 | 6.4 | 1.60 |
| TM7S | 100.0% | 10/10 | 7.2 | 1.68 |
| TM7SX | 100.0% | 10/10 | 6.8 | 1.66 |
| Toney_M10 | 100.0% | 10/10 | 7.6 | 1.75 |
| UR10 | 100.0% | 10/10 | 7.2 | 1.70 |
| UR10e | 100.0% | 10/10 | 7.6 | 1.74 |
| UR16e | 100.0% | 10/10 | 7.8 | 1.77 |
| UR20 | 100.0% | 10/10 | 6.8 | 1.66 |
| UR3 | 100.0% | 10/10 | 6.4 | 1.59 |
| UR30 | 100.0% | 10/10 | 6.8 | 1.66 |
| UR3e | 100.0% | 10/10 | 7.0 | 1.68 |
| UR5 | 100.0% | 10/10 | 7.0 | 1.70 |
| UR5e | 100.0% | 10/10 | 6.8 | 1.64 |

### ✅ All Robots Passed

All collaborative robots passed all test poses successfully!

---

## Test 2: Stress Test with Random Configurations

### Test Overview
- **Configurations Tested:** 100
- **Passed Tests:** 100
- **Failed Tests:** 0
- **Success Rate:** 100.0%
- **Average Solutions per Test:** 3.1
- **Average Time per Test:** 1.03 ms

### Edge Cases Tested
The stress test includes systematic edge case testing:
- **θ₁ = 180°:** Every 10th test forces joint 1 to 180°
- **θ₃ = 0°:** Every 15th test forces joint 3 to 0°
- **θ₃ = 180°:** Every 20th test forces joint 3 to 180°
- **θ₅ = 0°:** Every 25th test forces joint 5 to 0°

### ✅ All Tests Passed

All random configurations passed successfully, including edge cases!

## Technical Details

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
