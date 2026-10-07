# GMR Unitree G1 29-DoF evaluation

**Status:** REVIEW

- Frames: 623 at 30 FPS
- Processing rate: 196.9 FPS
- Mean matched-body position error: 7.84 cm
- Mean upper-body position error: 9.42 cm
- Mean lower-body position error: 6.91 cm
- Joint-limit violations: 0
- Velocity-limit violations: 0
- Maximum motor speed: 5.307 rad/s
- Maximum motor acceleration: 127.142 rad/s^2
- Frames over the diagnostic acceleration threshold: 21
- Minimum toe-body height: 0.119 m
- Self-collision contacts: 0
- Maximum waist speed: 1.594 rad/s

## Review reasons

- 21 frames exceed the 50 rad/s^2 diagnostic acceleration threshold; the peak is 127.1 rad/s^2 at frame 1 on right_elbow_joint. Smooth and re-evaluate before controller training or hardware use.

The acceleration review threshold is a conservative workflow diagnostic, not a
manufacturer-specified G1 actuator limit.

This is a kinematic reference-motion evaluation in MuJoCo, not a dynamically
controlled rollout and not authorization for physical-robot deployment.
