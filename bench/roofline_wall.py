"""The device read-bandwidth wall, three readings; the denominator for
bench/parity_rotation.py's efficiency column (run this first).

Uses the product's own probe (drinkme.probe.measure_bandwidth) unchanged.
The read figure (full reduction, every byte read once, one scalar written)
lands at verification/parity/wall.json (or the path given as argv[1]).
"""
import json
import os
import sys
import time
import traceback
from pathlib import Path


def main():
    import torch
    from drinkme.probe import measure_bandwidth
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required')
    out = Path(sys.argv[1] if len(sys.argv) > 1 else 'verification/parity/wall.json')
    free, total = torch.cuda.mem_get_info()
    props = torch.cuda.get_device_properties(0)
    report = dict(verdict='INCOMPLETE', device=torch.cuda.get_device_name(), torch=torch.__version__,
                  hip=getattr(torch.version, 'hip', None), cuda=getattr(torch.version, 'cuda', None),
                  # gcnArchName is the ROCm build's; a CUDA build has the capability
                  arch=getattr(props, 'gcnArchName', None) or f'sm_{props.major}{props.minor}',
                  mem_free_bytes_before=free, mem_total_bytes=total, readings=[])
    for i in range(3):
        t0 = time.time()
        r = measure_bandwidth()
        r['wall_seconds'] = time.time() - t0
        report['readings'].append(r)
        print('WALL', i, r['read_bytes_s'], 'read B/s', r['copy_bytes_s'], 'copy B/s', r['probe_bytes'], 'probe bytes', flush=True)
    reads = [r['read_bytes_s'] for r in report['readings']]
    report['read_bytes_s_median'] = int(sorted(reads)[1])
    report['read_bytes_s_min'] = int(min(reads))
    report['read_bytes_s_max'] = int(max(reads))
    report['verdict'] = 'PASS'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + '\n')
    print('ROOFLINE_WALL PASS median', report['read_bytes_s_median'], 'B/s', flush=True)


if __name__ == '__main__':
    status = 1
    try:
        main()
        status = 0
    except BaseException:
        print('ROOFLINE_WALL FAIL', flush=True)
        traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(status)
