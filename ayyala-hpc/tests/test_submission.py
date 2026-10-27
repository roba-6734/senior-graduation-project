"""GPU-free regression checks for submission and Slurm spool handling."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class SubmissionTests(unittest.TestCase):
    def test_isolated_submission_and_spooled_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            bundle = base / 'cloned bundle'
            shutil.copytree(ROOT, bundle)
            config = bundle / 'config.env'
            config.write_text(
                f'export UNITREE_WORKDIR="{base / "work"}"\n'
                'export NUM_ENVS=4096\nexport MAX_ITERATIONS=30000\n'
                'export SLURM_PARTITION=test\nexport SLURM_GRES=gpu:1\n'
            )
            bin_dir = base / 'bin'
            bin_dir.mkdir()
            recorder = bin_dir / 'sbatch'
            recorder.write_text(
                '#!/usr/bin/env python3\n'
                'import json, os, sys\n'
                'with open(os.environ["CALLS"], "a") as f:\n'
                '    f.write(json.dumps(sys.argv[1:]) + "\\n")\n'
                'print("123;cluster")\n'
            )
            recorder.chmod(0o755)
            calls = base / 'calls.jsonl'
            env = {**os.environ, 'PATH': f'{bin_dir}:{os.environ["PATH"]}',
                   'CALLS': str(calls)}
            import json
            for mode, expected in [('smoke', '512 50'), ('full', '4096 30000')]:
                calls.write_text('')
                subprocess.run(['bash', str(bundle / 'submit_pipeline.sh'), mode,
                                str(config)], env=env, check=True, capture_output=True)
                records = [json.loads(line) for line in calls.read_text().splitlines()]
                self.assertEqual(len(records), 3)
                self.assertIn('--dependency=afterok:123', records[1])
                self.assertIn('--dependency=afterok:123', records[2])
                exported = next(arg for arg in records[1] if arg.startswith('--export='))
                values = dict(item.split('=', 1) for item in exported.split(',')[1:])
                self.assertEqual(values['HPC_BUNDLE_DIR'], str(bundle))
                # Execute the real training configuration from a simulated Slurm spool.
                # Skip GPU/Conda calls, retaining path resolution and config loading.
                script = (bundle / 'train.sbatch').read_text().split('SEED=')[0]
                script = '\n'.join(line for line in script.splitlines() if line not in (
                    'activate_unitree_env', 'require_unitree_commit',
                    'require_file "$MOTION_NPZ"', 'check_gpu'))
                script += '\nprintf "%s %s\\n" "$NUM_ENVS" "$MAX_ITERATIONS"\n'
                spool = base / 'slurm_script'
                spool.write_text(script)
                result = subprocess.run(['bash', str(spool)], env={**env, **values},
                                        cwd=base, check=True, capture_output=True, text=True)
                self.assertEqual(result.stdout.strip(), expected)


if __name__ == '__main__':
    unittest.main()
