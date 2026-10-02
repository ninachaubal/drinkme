"""HTTP/SSE A/B of MTP suffix attention on one resident model.

Run with the existing interpreter and PYTHONPATH=src; never resolve the
environment. --pack-dir is read only. The test server uses a private port.
Alternates reference/native attention, keeps resident prefix slots enabled,
and clears them before each request. Separate synchronized phase timings
diagnose the cycle; they are not used in the throughput results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

import serve_drive


class Phases:
    def __init__(self, engine):
        self.engine = engine
        self.rows = []
        self.current = None

    def __enter__(self):
        import torch
        from drinkme.serving import mtp
        self.restore = []
        for obj, name, phase in ((mtp.Speculator, 'cycle', 'cycle'),
                                 (self.engine.mtp_head, 'draft', 'draft'),
                                 (mtp, 'forward_with_hidden', 'verify')):
            original = getattr(obj, name)
            def wrap(*a, _fn=original, _phase=phase, **kw):
                if _phase == 'cycle':
                    self.current = {}
                if self.current is None:
                    return _fn(*a, **kw)
                torch.cuda.synchronize()
                start = time.perf_counter()
                out = _fn(*a, **kw)
                torch.cuda.synchronize()
                self.current[_phase+'_ms'] = (time.perf_counter()-start)*1000
                if _phase == 'verify':
                    self.current['verify_rows'] = a[1].shape[1]
                if _phase == 'cycle':
                    self.rows.append(self.current)
                    self.current = None
                return out
            setattr(obj, name, wrap)
            self.restore.append((obj, name, original))
        return self

    def __exit__(self, *exc):
        for obj, name, original in reversed(self.restore):
            setattr(obj, name, original)

    def summary(self):
        rows = [r for r in self.rows[2:] if r['verify_rows'] == 5]
        return {k: statistics.median(r[k] for r in rows)
                for k in ('cycle_ms', 'draft_ms', 'verify_ms')} if rows else {}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pack-dir', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--runs', type=int, default=3)
    ap.add_argument('--tokens', type=int, default=48)
    ap.add_argument('--ctx', type=int, default=16384)
    ap.add_argument('--prompt-repeats', nargs='+', type=int, default=[0, 96, 384])
    ap.add_argument('--prompt-file', type=Path)
    ap.add_argument('--phases', action='store_true')
    ap.add_argument('--serial-check', action='store_true', help='also compare each transcript with serial decode')
    args = ap.parse_args()
    if args.runs < 1 or args.tokens < 2 or any(n < 0 for n in args.prompt_repeats):
        ap.error('runs must be positive, tokens >= 2, and prompt-repeats nonnegative')
    os.environ.update(DRINKME_SPEC='auto', DRINKME_SLOT_DIR='off', DRINKME_PREFIX_SLOTS='1')
    os.environ.pop('DRINKME_MTP_DEPTH', None)
    import torch
    from drinkme.serve import build_engine
    from drinkme.serving.http import start_server
    from drinkme.serving import suffix_attention
    assert torch.cuda.is_available(), 'REFUSING CPU measurement'
    torch.set_num_threads(4)
    meta_bytes = (Path(args.pack_dir)/'meta.json').read_bytes()
    meta = json.loads(meta_bytes)
    report = {'torch': torch.__version__, 'hip': torch.version.hip,
              'device': torch.cuda.get_device_name(), 'pack': args.pack_dir,
              'pack_meta_sha256': hashlib.sha256(meta_bytes).hexdigest(),
              'gpu_memory_before': torch.cuda.mem_get_info(), 'cases': [], 'complete': False,
              'ctx': args.ctx, 'tokens': args.tokens, 'runs': args.runs, 'threads': 4,
              'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'source_diff_sha256': hashlib.sha256(subprocess.check_output(['git', 'diff', 'HEAD'])).hexdigest(),
              'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                (Path('src/drinkme/serving/mtp.py'),
                                 Path('src/drinkme/serving/suffix_attention.py'),
                                 Path('src/drinkme/serving/kvcache.py'))},
              'environment': {k: v for k, v in os.environ.items() if k.startswith('DRINKME_')
                              or k in ('TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL', 'HF_HUB_OFFLINE')}}
    def save():
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2)+'\n')
    save()
    eng = build_engine(meta['hfRepo'], meta.get('revision'), args.pack_dir, stock=False, ctx=args.ctx)
    assert eng.mtp_head is not None
    base = eng.model.get_decoder()
    assert type(base).__name__ == 'Qwen3_5TextModel'
    assert all(layer.self_attn.config._attn_implementation == suffix_attention.NAME
               for layer in base.layers if hasattr(layer, 'self_attn'))
    srv = start_server(eng, '127.0.0.1', 0)
    port = srv.server_address[1]
    report['health'] = serve_drive.health(port)
    task = args.prompt_file.read_text() if args.prompt_file else serve_drive.PROMPT
    try:
        for repeats in args.prompt_repeats:
            serve_drive.PROMPT = ('Background notes for this task:\n' +
                'The river flows past the old oak tree. The boat is red and the owl is grey.\n'*repeats +
                '\nTask:\n' + task if repeats else task)
            case = {'repeats': repeats, 'arms': {'reference': [], 'native': []}, 'phases': {}}
            case['prompt_sha256'] = hashlib.sha256(serve_drive.PROMPT.encode()).hexdigest()
            for arm in case['arms']:
                os.environ[suffix_attention.ENV] = '1' if arm == 'native' else '0'
                eng.reset_prefix_cache()
                serve_drive.request(port, 8)
            for run in range(args.runs):
                order = ('reference', 'native') if run % 2 == 0 else ('native', 'reference')
                for arm in order:
                    os.environ[suffix_attention.ENV] = '1' if arm == 'native' else '0'
                    eng.reset_prefix_cache()
                    before = serve_drive.metrics(port)
                    row = serve_drive.request(port, args.tokens)
                    after = serve_drive.metrics(port)
                    row['mtp'] = {k: after[k]-before.get(k, 0) for k in after if k.startswith('drinkme_mtp_')}
                    assert row['completion_tokens'] == args.tokens
                    assert row['mtp']['drinkme_spec_proposed_tokens_total'] > 0
                    case['arms'][arm].append(row)
                    print(f'{repeats=} {run=} {arm}: {row["decode_tok_s"]:.3f} tok/s', flush=True)
            case['same_text'] = len({r['text'] for rows in case['arms'].values() for r in rows}) == 1
            case['median_tok_s'] = {arm: statistics.median(r['decode_tok_s'] for r in rows)
                                    for arm, rows in case['arms'].items()}
            if args.phases:
                for arm in case['arms']:
                    os.environ[suffix_attention.ENV] = '1' if arm == 'native' else '0'
                    eng.reset_prefix_cache()
                    with Phases(eng) as p:
                        serve_drive.request(port, args.tokens)
                    case['phases'][arm] = {'median': p.summary(), 'rows': p.rows}
            if args.serial_check:
                os.environ['DRINKME_SPEC'] = 'off'
                eng._spec_plan = None
                eng._mtp_depth = None
                eng.reset_prefix_cache()
                serial = serve_drive.request(port, args.tokens)
                case['serial'] = serial
                case['same_as_serial'] = all(serial['text'] == row['text']
                                             for rows in case['arms'].values() for row in rows)
                os.environ['DRINKME_SPEC'] = 'auto'
                eng._spec_plan = None
                eng._mtp_depth = None
            report['cases'].append(case)
            save()
            print(f'CASE {repeats}: {case["median_tok_s"]}, {case["same_text"]=}, phases=' +
                  str({a: r['median'] for a, r in case['phases'].items()}), flush=True)
        assert (Path(args.pack_dir)/'meta.json').read_bytes() == meta_bytes
        report['complete'] = True
        save()
    finally:
        srv.shutdown()
        srv.server_close()
    print('VERDICT: PASS measurements complete (see same_text for transcript agreement)', flush=True)


if __name__ == '__main__':
    status = 0
    try:
        main()
    except SystemExit as exc:
        status = exc.code if isinstance(exc.code, int) else 1
    except BaseException:
        traceback.print_exc()
        status = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(status)  # TheRock may mask failures during normal interpreter exit.
