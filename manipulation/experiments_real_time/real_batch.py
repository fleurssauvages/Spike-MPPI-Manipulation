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
    p.add_argument('--resume', action='store_true',
                   help='keep every finished trial, failed or not, and continue with the next seed')
    p.add_argument('--stop-on-error', action='store_true')
    p.add_argument('--max-consecutive-aborts', type=int, default=3,
                   help='a safety-aborted trial is kept and the batch continues, '
                        'unless this many trials in a row abort')
    return p


def _is_finished_trial(path: Path) -> bool:
    """A complete hardware recording: success, safety abort, or Ctrl-C (a failure)."""
    if missing_analysis_keys(path):
        return False
    with np.load(path, allow_pickle=False) as record:
        return 'hardware_mode' in record.files


def _move_aside(path: Path) -> Path:
    """Move a recording into replaced/ instead of overwriting it.

    A subfolder keeps it out of the run_*_seed_*.npz glob used by the analysis.
    """
    folder = path.parent / 'replaced'
    folder.mkdir(exist_ok=True)
    k = 1
    while (dest := folder / f'{path.stem}.{k}{path.suffix}').exists():
        k += 1
    path.rename(dest)
    return dest


def main():
    args = make_batch_parser().parse_args()
    if args.repeats < 1:
        raise ValueError('--repeats must be >=1')
    if args.max_consecutive_aborts < 1:
        raise ValueError('--max-consecutive-aborts must be >=1')
    args.viewer = False
    args.analysis_trace = True
    args.no_save = False
    start_seed = args.seed if args.seed_start is None else args.seed_start
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = {k: v for k, v in vars(args).items() if isinstance(v, (str, int, bool, float))}
    (output / 'config.json').write_text(json.dumps(config, indent=2, sort_keys=True)+'\n')
    rows = []
    # Errors from earlier launches are kept; new ones are appended.
    errors_path = output / 'errors.json'
    errors = json.loads(errors_path.read_text()) if errors_path.is_file() else []
    previous_errors = len(errors)
    consecutive_aborts = 0
    for i in range(args.repeats):
        args.seed = int(start_seed) + i
        dest = output / f'run_{i:03d}_seed_{args.seed}.npz'
        args.save = str(dest)
        stop_batch = None
        interrupted = False
        try:
            reused = args.resume and dest.is_file() and _is_finished_trial(dest)
            if not reused and dest.is_file():
                moved = _move_aside(dest)
                print(f'[{i+1}/{args.repeats}] seed={args.seed}: previous recording moved to {moved}', flush=True)
            if reused:
                print(f'[{i+1}/{args.repeats}] seed={args.seed}: already done, keeping it', flush=True)
            else:
                pass
                result = run(args)
                if result['stop_reason'] == 'keyboard_interrupt':
                    # A Ctrl-C'd trial is kept as a failure; the batch stops after
                    # summarizing it. Interrupted before any command: nothing to keep.
                    if int(result.get('steps', 0)) == 0 or not dest.is_file():
                        break
                    errors.append({'trial':i, 'seed':args.seed,
                                   'error':result['stop_detail'], 'stop_reason':result['stop_reason']})
                    interrupted = True
                elif result['stop_reason'] == 'safety_abort':
                    errors.append({'trial':i, 'seed':args.seed,
                                   'error':result['stop_detail'], 'stop_reason':result['stop_reason']})
                    consecutive_aborts += 1
                    # Aborting before any MPPI command (homing, feedback) is a setup
                    # problem, not a trial outcome; repeated aborts likely are too.
                    if int(result.get('steps', 0)) == 0 or not dest.is_file():
                        raise RuntimeError('Hardware safety abort before the first MPPI command; '
                                           'stopping entire batch')
                    if consecutive_aborts >= args.max_consecutive_aborts:
                        # Summarize this kept trial first, then stop below.
                        stop_batch = (f'Hardware safety abort {consecutive_aborts} trials in a row; '
                                      'stopping entire batch')
                    print(f'[{i+1}/{args.repeats}] seed={args.seed}: safety abort '
                          f'({result["stop_detail"]}); kept as a failed trial, '
                          f'{"stopping" if stop_batch else "continuing"}', flush=True)
                else:
                    consecutive_aborts = 0
            missing = missing_analysis_keys(dest)
            if missing:
                raise RuntimeError(f'Missing analysis NPZ fields: {missing}')
            with np.load(dest, allow_pickle=False) as record:
                if len(record['controls']) == 0:
                    if reused:  # kept failure that aborted before its first command
                        continue
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
            if interrupted:
                break
            if stop_batch:
                raise RuntimeError(stop_batch)
        except (KeyboardInterrupt, EOFError):
            pass
            break
        except Exception as exc:
            if not (len(errors) > previous_errors and errors[-1].get('trial') == i):
                errors.append({'trial':i,'seed':args.seed,'error':str(exc),
                               'traceback':traceback.format_exc()})
            pass
            # These safety aborts end the hardware batch, regardless of stop-on-error.
            if 'Hardware safety abort' in str(exc) or args.stop_on_error or args.execute:
                break
        finally:
            _write_csv(output/'summary.csv', rows)
            errors_path.write_text(json.dumps(errors,indent=2)+'\n')
    pass


if __name__ == '__main__':
    main()
