# Smoothed GMR v2 evaluation

**Status:** PASS

| Metric | Raw v1 | Smoothed v2 |
|---|---:|---:|
| Mean matched-body error | 7.84 cm | 7.84 cm |
| Mean upper-body error | 9.42 cm | 9.42 cm |
| Maximum motor speed | 5.31 rad/s | 2.93 rad/s |
| P99 motor acceleration | 32.18 rad/s^2 | 7.47 rad/s^2 |
| Maximum motor acceleration | 127.14 rad/s^2 | 35.47 rad/s^2 |
| Maximum root linear acceleration | 12.39 m/s^2 | 4.33 m/s^2 |
| Maximum root angular acceleration | 51.69 rad/s^2 | 20.53 rad/s^2 |
| Maximum contact-segment displacement | 3.06 cm | 3.06 cm |

Mean absolute joint adjustment: 0.00039 rad.

## Acceptance checks

- PASS — all 623 frames preserved
- PASS — zero joint limit violations
- PASS — zero velocity limit violations
- PASS — maximum acceleration below 50 rad s2
- PASS — root linear acceleration below 6 m s2
- PASS — root angular acceleration below 50 rad s2
- PASS — mean position error at most 8 5 cm
- PASS — mean error increase below 1 cm
- PASS — mean upper body error at most 10 cm
- PASS — zero self collision contacts
- PASS — no toe body ground penetration
- PASS — contact displacement not increased over 5 mm
- PASS — mean contact foot speed not increased over 2 mm s
- PASS — p95 contact foot speed not increased over 1 cm s
- PASS — maximum contact foot speed not increased over 2 cm s

The 40 rad/s^2 smoothing constraint and 50 rad/s^2 acceptance threshold are
conservative project diagnostics, not manufacturer-specified actuator limits.
This is still a kinematic reference. PASS means it is ready for dynamic
tracking tests in simulation, not deployment on a physical robot.
