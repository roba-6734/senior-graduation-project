# Ayyala G1 tracking on an A5000 Slurm cluster

This bundle trains the 29-DoF smoothed GMR v2 reference with Unitree's official
MuJoCo tracking task. It is pinned to `unitreerobotics/unitree_rl_mjlab` commit
`1425b15f73bd4095f0df53709d7c389c3eb9e790` so the exporter, evaluator, and
upstream APIs stay synchronized.

The workflow is:

1. convert the validated 30 Hz, 36-column CSV to Unitree's 50 Hz tracking NPZ;
2. train `Unitree-G1-Tracking-No-State-Estimation` with PPO;
3. evaluate one complete 20.7-second reference without automatic resets;
4. save per-frame tracking metrics, a summary, and a ghost/reference video.

This is simulation work. A successful job is not authorization to run the
motion on a physical G1.

## One-time cluster setup

The cluster needs Ubuntu-compatible NVIDIA drivers, an A5000 allocation,
Conda, Git, and network access while installing. Unitree currently recommends
Python 3.11, an NVIDIA GPU, and driver 550 or later.

Copy the whole project to a shared filesystem visible from login and compute
nodes. Then configure paths and scheduler fields:

```bash
cd /shared/path/to/seniorproject/hpc/unitree_mjlab
cp config.env.example config.env
# Edit PROJECT_ROOT, UNITREE_WORKDIR, partition, account, and GRES.
```

Load the cluster's Conda module if necessary, then install the pinned software
without administrator privileges:

```bash
module load miniconda  # use the equivalent command on your cluster
./setup_env.sh config.env
```

`UNITREE_WORKDIR` should be scratch or project storage, not a small home quota.
The setup script never replaces an existing Unitree checkout at a different
commit; it stops so the mismatch can be reviewed.

## Submit a smoke test first

```bash
./submit_pipeline.sh smoke config.env
```

The smoke configuration uses 512 environments and 50 iterations. It is only an
integration check: conversion must pass, training must create a checkpoint,
and evaluation must create metrics and video. Its policy is not expected to
track the dance well.

Inspect:

```text
<UNITREE_WORKDIR>/slurm_logs/
<PROJECT_ROOT>/hpc_results/<run-tag>/tracking_metrics.md
<PROJECT_ROOT>/hpc_results/<run-tag>/ayyala_policy_vs_reference.mp4
```

If CUDA runs out of memory, reduce `SMOKE_NUM_ENVS` or `NUM_ENVS`. Reasonable
A5000 fallbacks are 2048, 1024, then 512 environments.

## Submit full training

After the smoke pipeline succeeds:

```bash
./submit_pipeline.sh full config.env
```

The provided starting point is 4096 environments, seed 42, and 30,000 PPO
iterations. Cluster limits and observed learning curves may require changing
the wall time or iteration count in `config.env`/`train.sbatch`. The final
checkpoint is recorded under:

```text
<UNITREE_WORKDIR>/state/<run-tag>/checkpoint.txt
```

For a research comparison, train at least three seeds for GMR and ProtoMotion
using the same task, rewards, number of environments, iterations, and evaluator.
Only the reference motion should change.

## What the evaluator reports

`tracking_metrics.json` and `tracking_metrics.md` report dynamic
reference-to-simulation quantities:

- MPKPE and root-relative MPKPE;
- end-effector position and orientation error;
- root position/orientation error;
- joint-position and joint-velocity error;
- actuator force and saturation frequency;
- action clipping, joint-limit violations, and termination-equivalent failures;
- whether the policy survived the complete reference.

`tracking_timeseries.npz` contains the per-frame values. If the local GMR v2
report is present, the summary also carries the separate human-to-robot
retargeting error. Do not add those errors together: they measure different
pipeline stages.

## Manual job submission

The dependency chain is normally safer, but each stage can be submitted
manually after exporting `HPC_CONFIG` and `RUN_TAG`:

```bash
export HPC_CONFIG=/shared/path/to/seniorproject/hpc/unitree_mjlab/config.env
export RUN_TAG=ayyala_manual_seed42
sbatch prepare_motion.sbatch
sbatch train.sbatch
sbatch evaluate.sbatch
```

When submitting manually, add `afterok` dependencies yourself and ensure the
cluster's A5000 GRES syntax is correct. Scheduler resource names vary between
clusters; `gpu:a5000:1` in the example is not universal.

## Upstream references

- Unitree RL MjLab: <https://github.com/unitreerobotics/unitree_rl_mjlab>
- Official setup guide: <https://github.com/unitreerobotics/unitree_rl_mjlab/blob/main/doc/setup_en.md>
- Official CSV converter: <https://github.com/unitreerobotics/unitree_rl_mjlab/blob/main/scripts/csv_to_npz.py>

