# Ayyala source-motion validation

Run the dependency-light validator against the supplied directory of NumPy
arrays:

```bash
python3 tools/validate_smpl_source.py \
  ayyala_visible_feet_long_static_pre_inv_rot1.npz \
  --output-dir validation_output \
  --stem ayyala_visible_feet_long_static_pre_inv_rot1
```

The command creates:

- a real SMPL `.npz` archive;
- a GMR-compatible SMPL-X `.npz` archive using the same mapping as GMR's
  `scripts/smpl_to_smplx.py`;
- a JSON and HTML validation report;
- per-frame motion metrics and diagnostic SVG plots.

The SMPL-X source has now been rendered with the licensed model and aligned to
the original video. The coordinate system was confirmed as Z-up with Y as
camera depth; the validated archive below is the input to the GMR baseline.

## GMR 29-DoF simulation baseline

The source/video validation has resolved the coordinate warning: Z is vertical
and Y is camera depth. Run the headless GMR baseline with:

```bash
MUJOCO_GL=egl .venv-source/bin/python tools/run_gmr_retarget.py \
  --motion validation_output/ayyala_visible_feet_long_static_pre_inv_rot1_smplx.npz \
  --model-root third_party/GMR/assets/body_models \
  --gmr-repo third_party/GMR \
  --output-dir gmr_output
```

This target is the 29-DoF G1 and is for simulation evaluation only until the
physical robot configuration is confirmed.

The current 623-frame run is classified **REVIEW**, not deployment-ready. It
has no joint-limit, configured velocity-limit, self-collision, or toe-ground
penetration failures, but 21 frames exceed a conservative 50 rad/s²
acceleration diagnostic threshold. That threshold is a project review rule,
not a Unitree actuator specification. Smooth those discontinuities and test a
tracking controller in simulation before considering hardware.

Main outputs:

- `gmr_output/ayyala_gmr_unitree_g1_29dof.mp4`: visual preview;
- `gmr_output/ayyala_gmr_unitree_g1_29dof.pkl`: GMR motion reference;
- `gmr_output/gmr_evaluation.md`: readable evaluation summary;
- `gmr_output/gmr_evaluation.json`: complete machine-readable metrics;
- `gmr_output/ayyala_gmr_diagnostics.npz`: per-frame diagnostic arrays.

## Contact-aware smoothed v2 reference

Create the training-candidate reference from the raw GMR qpos with:

```bash
MUJOCO_GL=egl .venv-source/bin/python tools/smooth_gmr_motion.py \
  --raw-qpos gmr_output/ayyala_gmr_qpos_wxyz.npy \
  --raw-video gmr_output/ayyala_gmr_unitree_g1_29dof.mp4 \
  --motion validation_output/ayyala_visible_feet_long_static_pre_inv_rot1_smplx.npz \
  --model-root third_party/GMR/assets/body_models \
  --output-dir gmr_output_v2
```

The v2 pass jointly smooths pelvis translation/orientation and all 29 joints
inside detected discontinuity windows. It constrains joint, velocity,
acceleration, and contact-foot displacement while minimizing deviation from
the raw reference. The current result passes all kinematic acceptance checks:
maximum motor acceleration falls from 127.14 to 35.47 rad/s² without degrading
the 7.84 cm mean matched-body error or foot-slip statistics.

Main v2 outputs:

- `gmr_output_v2/ayyala_gmr_unitree_g1_29dof_v2_smoothed.pkl`;
- `gmr_output_v2/ayyala_gmr_unitree_g1_29dof_v2_smoothed.mp4`;
- `gmr_output_v2/ayyala_gmr_v1_vs_v2.mp4`;
- `gmr_output_v2/gmr_v2_evaluation.md` and `.json`;
- `gmr_output_v2/ayyala_gmr_v2_diagnostics.npz`.

A v2 PASS means the reference can proceed to dynamic controller tracking in
simulation. It is not approval for physical-robot deployment.
