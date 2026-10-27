# Ayyala G1 HPC training

Standalone Slurm bundle for training a Unitree G1 29-DoF tracking policy on the
smoothed Ayyala reference. No files from the parent seniorproject repository
are needed. The motion CSV, checksum manifest, evaluation tools, and optional
kinematic retargeting report are included. SMPL body models and original video
are not required.

## Clone and configure

Clone your new repository onto storage visible at the same path on login and
compute nodes. Run all commands below from its root directory.

```bash
git clone <YOUR_REPOSITORY_URL> ayyala-hpc
cd ayyala-hpc
cp config.env.example config.env
nano config.env
```

Set `UNITREE_WORKDIR` to a writable absolute shared scratch/project path with
room for the environment, upstream checkout, and checkpoints. Set
`SLURM_PARTITION`, `SLURM_ACCOUNT`, and `SLURM_GRES` to your allocation's values.
The example GPU request is for one A5000; your cluster may use another name.
Discover partitions with `sinfo -o "%P %a %l %G"`. Avoid commas in repository or
config paths because Slurm uses commas to separate exported variables.

## Install once

Requirements: Bash, Git, Conda, Slurm, an NVIDIA GPU allocation, and internet
access for installation. Load your cluster's Conda module if required
(`module avail` lists modules), or set `CONDA_EXE` in the config. Install on a
node where your HPC permits package installation:

```bash
bash setup_env.sh config.env
```

Setup creates Python 3.11 and installs Unitree RL MjLab pinned to commit
`1425b15f73bd4095f0df53709d7c389c3eb9e790`. It records Conda and pip package
versions under `UNITREE_WORKDIR`. Transitive Python dependencies are resolved
at installation time, not locked by this bundle. A login node need not expose
a GPU; each scheduled job checks GPU access. The Conda executable must also be
available on compute nodes; setting its absolute path is useful on clusters
whose module environment does not carry over.

## Run a smoke test

```bash
bash submit_pipeline.sh smoke config.env
squeue -u "$USER"
```

This submits conversion, training, and evaluation with `afterok` dependencies.
Conversion validates the 623-frame, 30 Hz CSV and creates the 50 Hz tracking
NPZ. Smoke training uses 512 environments and 50 iterations, independently of
the full-training settings in `config.env`. Expect a poorly trained policy;
the purpose is to verify checkpoint, metrics, and video generation.

The submission prints three job IDs and a run tag. Logs are under
`<UNITREE_WORKDIR>/slurm_logs/`. To inspect completed jobs, substitute the printed
IDs:

```bash
sacct -j <CONVERT_ID>,<TRAIN_ID>,<EVAL_ID> --format=JobID,State,ExitCode,Elapsed
```

If an upstream job fails, dependent jobs cannot run; inspect its log and cancel
those pending jobs with `scancel <TRAIN_ID> <EVAL_ID>` before resubmitting.

## Full training

After all three smoke stages complete successfully:

```bash
bash submit_pipeline.sh full config.env
```

Defaults are 4096 environments, 30,000 PPO iterations, and seed 42. If CUDA runs
out of memory, lower `NUM_ENVS` to 2048, 1024, or 512; for smoke use
`SMOKE_NUM_ENVS`. Adjust the `#SBATCH` CPU, memory, and time requests to your
cluster's limits before submission. The same training allocation is requested
for smoke and full runs; you may shorten its time for the smoke test.

| Stage | CPUs | RAM | Time limit | GPUs |
| --- | ---: | ---: | --- | ---: |
| Convert | 4 | 16 GB | 30 minutes | 1 |
| Train | 12 | 64 GB | 48 hours | 1 |
| Evaluate | 8 | 32 GB | 1 hour | 1 |

These are requests, not runtime predictions. Checkpoints are written during
training, but the pipeline does not automatically resume timed-out jobs.
Avoid concurrent pipelines sharing one work directory: conversion writes the
same motion NPZ. Use separate `UNITREE_WORKDIR` values for concurrent runs.

## Results

By default, results are in `hpc_results/<run-tag>/`:

- `tracking_metrics.md` and `.json`: tracking errors, actuator saturation,
  limit violations, and whether the policy survived the complete motion;
- `tracking_timeseries.npz`: per-frame measurements;
- `ayyala_policy_vs_reference.mp4`: policy/reference comparison.

The final checkpoint path is stored in
`<UNITREE_WORKDIR>/state/<run-tag>/checkpoint.txt`; checkpoints themselves live
under the upstream checkout's `logs/rsl_rl/g1_tracking/` directory. Copy valuable
results and checkpoints out before the cluster's scratch cleanup deadline.

Kinematic human-to-robot retargeting errors from `reports/` and dynamic policy
tracking errors describe separate stages. Simulation success does not establish
physical-robot readiness.

## Publish this folder as your own repository

Run these commands locally from this folder, after creating an empty remote:

```bash
git init -b main
git add .
git commit -m "Add standalone Ayyala HPC training bundle"
git remote add origin <YOUR_REPOSITORY_URL>
git push -u origin main
```

Local cluster configuration and generated results are ignored by Git. This
bundle downloads the upstream framework during setup instead of vendoring it.
Upstream: https://github.com/unitreerobotics/unitree_rl_mjlab

## Local checks

```bash
python3 -m unittest discover -s tests -v
```

These checks exercise Slurm submission and spooled-script configuration without
a GPU. Actual conversion, training, and rendering must be checked on the HPC.
