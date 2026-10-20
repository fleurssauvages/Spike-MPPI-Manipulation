"""Repeated real FR3 reaching runs; writes original batch summary.csv and NPZ keys.

The --execute flag authorizes all requested trials without repeated prompts.
Each run independently performs joint-position homing using the same PC1 position P controller.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback

import numpy as np

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from manipulation.franka_reaching.batch import missing_analysis_keys, summarize_recording, _write_csv
from manipulation.experiments_real_time.real_run import make_parser, run


def make_batch_parser() -> argparse.ArgumentParser:
    p = make_parser()
    p.description = 'Real FR3 sampled-reaching experiment batch at 100 Hz by default'
    p.add_argument('--repeats', type=int, default=100)
    p.add_argument('--seed-start', type=int, default=None)
    p.add_argument('--output-dir', default='results/reaching_real_batch')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--stop-on-error', action='store_true')
    return p


def main():
    args = make_batch_parser().parse_args()
    if args.repeats < 1:
        raise ValueError('--repeats must be >=1')
    args.viewer = False
    args.analysis_trace = True
    args.no_save = False
    start_seed = args.seed if args.seed_start is None else args.seed_start
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = {k: v for k, v in vars(args).items() if isinstance(v, (str, int, bool, float))}
    (output / 'config.json').write_text(json.dumps(config, indent=2, sort_keys=True)+'\n')
    rows, errors = [], []
    for i in range(args.repeats):
        args.seed = int(start_seed) + i
        dest = output / f'run_{i:03d}_seed_{args.seed}.npz'
        args.save = str(dest)
        try:
            reused = False
            if args.resume and dest.is_file() and not missing_analysis_keys(dest):
                with np.load(dest, allow_pickle=False) as existing:
                    reused = ('hardware_mode' in existing.files and
                              bool(existing['success']))
            if not reused:
                pass
                result = run(args)
                if result['stop_reason'] == 'safety_abort':
                    errors.append({'trial':i, 'seed':args.seed,
                                   'error':result['stop_detail'], 'stop_reason':result['stop_reason']})
                    raise RuntimeError('Hardware safety abort; stopping entire batch')
                if result['stop_reason'] == 'keyboard_interrupt':
                    pass
                    break
            missing = missing_analysis_keys(dest)
            if missing:
                raise RuntimeError(f'Missing analysis NPZ fields: {missing}')
            with np.load(dest, allow_pickle=False) as record:
                if len(record['controls']) == 0:
                    raise RuntimeError('Zero-command run cannot be summarized')
            row = summarize_recording(dest)
            row['trial'] = i
            with np.load(dest, allow_pickle=False) as record:
                row['stop_reason'] = str(record['stop_reason'])
                row['experiment_time_s'] = float(record['experiment_time_s'])
                row['commands_enabled'] = bool(record['commands_enabled'])
                row['max_loop_ms'] = float(np.nanmax(record['loop_ms']))
                row['max_tick_lateness_ms'] = float(np.nanmax(record['tick_lateness_ms']))
                row['mean_predicted_step_m'] = float(np.nanmean(record['predicted_step_m']))
            rows.append(row)
            pass
        except (KeyboardInterrupt, EOFError):
            pass
            break
        except Exception as exc:
            if not (errors and errors[-1].get('trial') == i):
                errors.append({'trial':i,'seed':args.seed,'error':str(exc),
                               'traceback':traceback.format_exc()})
            pass
            # Safety abort ALWAYS ends hardware batch, regardless of stop-on-error.
            if 'Hardware safety abort' in str(exc) or args.stop_on_error or args.execute:
                break
        finally:
            _write_csv(output/'summary.csv', rows)
            (output/'errors.json').write_text(json.dumps(errors,indent=2)+'\n')
    pass


if __name__ == '__main__':
    main()
