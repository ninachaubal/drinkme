"""Offline ROCm GEMM selection for the Qwen 27B's dominant 8K/32K projections.

Uses the existing environment. Run with the GPU benchmark lock; tuning can take
several minutes per shape. Writes a PyTorch TunableOp cache and a JSON receipt.
Only matching shapes use the saved choices; other shapes keep default dispatch.
Load the cache with tuning DISABLED when serving. It is device/runtime-specific,
so regenerate after hardware or runtime changes. No model or weights are changed.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

os.environ.update(OMP_NUM_THREADS='4', HF_HUB_OFFLINE='1')

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--tuning-file', type=Path, required=True)
    ap.add_argument('--from-report', type=Path, help='Export a completed report from this exact device/runtime')
    args = ap.parse_args()
    # Environment overrides take precedence over TunableOp's Python controls.
    # This instrument owns selection and keeps searches outside timed samples.
    for name in list(os.environ):
        if name.startswith('PYTORCH_TUNABLEOP_'):
            os.environ.pop(name)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.tuning_file.parent.mkdir(parents=True, exist_ok=True)
    import torch
    import torch.nn.functional as F
    from torch.cuda import tunable
    assert torch.cuda.is_available() and torch.version.hip
    torch.set_num_threads(4)
    torch.manual_seed(41)
    original = torch.backends.cuda.preferred_blas_library()
    report = dict(torch=torch.__version__, device=torch.cuda.get_device_name(),
                  initial_blas=str(original), cases=[], complete=False)
    def save():
        args.out.write_text(json.dumps(report, indent=2)+'\n')
    tunable.set_filename(str(args.tuning_file.resolve()))
    tunable.set_max_tuning_duration(10)
    tunable.set_max_tuning_iterations(2)
    tunable.tuning_enable(False)
    tunable.enable(False)
    def export(report):
        assert report['complete'] and report['torch'] == torch.__version__
        assert report['device'] == torch.cuda.get_device_name()
        results = report['cases'][-1]['tuning_results']
        validators = list(tunable.get_validators())
        with args.tuning_file.open('w', newline='') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerows([('Validator', *row) for row in validators])
            writer.writerows(results)
        assert tunable.read_file(str(args.tuning_file.resolve())), 'Cache validation failed'
        assert {tuple(r[:3]) for r in results} <= {tuple(r[:3]) for r in tunable.get_results()}
        report['validators'] = validators
        report['tuning_file'] = str(args.tuning_file.resolve())
        save()
        print('VERDICT: PASS exported and read back version-checked GEMM cache', flush=True)
    if args.from_report:
        report = json.loads(args.from_report.read_text())
        export(report)
        return
    with torch.inference_mode():
        for m, n, k in [(8192,17408,5120), (8192,5120,17408),
                         (32768,17408,5120), (32768,5120,17408)]:
            x = torch.randn(m,k,device='cuda',dtype=torch.bfloat16)
            w = torch.randn(n,k,device='cuda',dtype=torch.bfloat16)
            row = dict(m=m,n=n,k=k,times={}, errors={})
            def call(): return F.linear(x,w)
            # Small FP64 oracle over complete dot products, not truncated K.
            oracle = F.linear(x[:8].double(), w[:64].double())
            for name in ('default', 'cublas', 'cublaslt', 'tuned'):
                try:
                    tunable.enable(name == 'tuned')
                    torch.backends.cuda.preferred_blas_library('default' if name == 'tuned' else name)
                    start = time.perf_counter()
                    if name == 'tuned': tunable.tuning_enable(True)
                    y = call()
                    torch.cuda.synchronize()
                    if name == 'tuned':
                        tunable.tuning_enable(False)
                        row['tuning_seconds'] = time.perf_counter()-start
                        row['tuning_results'] = tunable.get_results()
                    err = (y[:8,:64].double()-oracle)
                    rel_rms = (err.square().mean()/oracle.square().mean()).sqrt().item()
                    assert rel_rms < .004, (name,rel_rms)
                    row['errors'][name] = dict(relative_rms=rel_rms, max_abs=err.abs().max().item())
                    del y
                    for _ in range(2): call()
                    samples=[]
                    for _ in range(5):
                        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        a.record(); y=call(); b.record(); b.synchronize()
                        samples.append(a.elapsed_time(b)); del y
                    row['times'][name] = samples
                    print(m,n,k,name,statistics.median(samples),'ms',flush=True)
                except RuntimeError as exc:
                    row['errors'][name] = str(exc)
                    print(name,'UNAVAILABLE',str(exc),flush=True)
                finally:
                    tunable.tuning_enable(False)
                    tunable.enable(False)
            row['median_ms'] = {name:statistics.median(v) for name,v in row['times'].items()}
            report['cases'].append(row); save()
            del x,w,oracle
    torch.backends.cuda.preferred_blas_library(original)
    report['complete'] = True
    save()
    export(report)
    print('VERDICT: PASS offline GEMM candidate experiment complete',flush=True)

if __name__ == '__main__':
    status = 0
    try:
        main()
    except SystemExit as exc:
        status = exc.code if isinstance(exc.code, int) else 1
    except BaseException:
        traceback.print_exc()
        status = 1
    sys.stdout.flush(); sys.stderr.flush(); os._exit(status)
