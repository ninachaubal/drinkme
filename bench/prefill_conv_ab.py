"""Cold HTTP/SSE prefill A/B over one resident model, serial and MTP.

Use the existing interpreter with PYTHONPATH=src and hold the GPU benchmark
lock. No environment resolution. The pack is read only, the server uses an
ephemeral localhost port, and the prefix cache is reset before each request.
--candidate-blas optionally measures a BLAS preference with the convolution.
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
import traceback

import serve_drive


def exact_prompt(engine, target, task, tokens):
    from drinkme.serving.engine import GenerationRequest, SampleParams

    filler = engine.tok.encode(
        'The river flows past the old oak tree. The boat is red and the owl is grey.\n'
        * (target // 16 + 10), add_special_tokens=False)
    n = max(0, target - 86)
    for _ in range(16):
        prompt = 'Background notes for this task:\n' + engine.tok.decode(filler[:n]) + '\nTask:\n' + task
        req = GenerationRequest([{'role': 'user', 'content': prompt}],
                                SampleParams(temperature=0, max_tokens=tokens),
                                template_kwargs={'enable_thinking': False})
        actual = engine.count_tokens(req)
        if actual == target:
            return prompt
        n += target - actual
        assert 0 <= n <= len(filler), (target, actual, n)
    raise AssertionError(f'Could not build exact {target}-token prompt: {actual}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pack-dir', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--contexts', nargs='+', type=int, default=[128, 8192, 32768])
    ap.add_argument('--modes', nargs='+', choices=['off', 'auto'], default=['off', 'auto'])
    ap.add_argument('--runs', type=int, default=3)
    ap.add_argument('--long-runs', type=int, default=1, help='Repeats for contexts >= 32K')
    ap.add_argument('--tokens', type=int, default=64)
    ap.add_argument('--prompt-file', type=Path)
    ap.add_argument('--candidate-blas', choices=['cublas', 'cublaslt'])
    ap.add_argument('--tuning-file', type=Path, help='Use an existing TunableOp cache on the candidate only')
    ap.add_argument('--reference-conv-fla', action='store_true', help='Keep FLA convolution in both arms to isolate GEMM selection')
    args = ap.parse_args()
    if min(args.contexts) < 128 or min(args.runs, args.long_runs) < 1 or args.tokens < 2:
        ap.error('contexts >= 128, repeats >= 1, and output tokens >= 2 are required')
    # Inherited environment overrides would defeat the per-arm Python controls.
    for name in list(os.environ):
        if name.startswith('PYTORCH_TUNABLEOP_'):
            os.environ.pop(name)
    os.environ.update(DRINKME_SPEC='auto' if 'auto' in args.modes else 'off',
                      DRINKME_DELTANET_CONV='fla', DRINKME_SLOT_DIR='off',
                      DRINKME_NO_AUTO_DEPS='1', DRINKME_PREFIX_SLOTS='1')
    os.environ.pop('DRINKME_MTP_DEPTH', None)
    import torch
    from torch.cuda import tunable
    from drinkme.serve import DEFAULT_PORT, build_engine
    from drinkme.serving.http import start_server
    from drinkme.serving import deltanet_conv

    assert torch.cuda.is_available(), 'REFUSING CPU measurement'
    torch.set_num_threads(4)
    # Timed requests must never discover/tune new GEMM solutions.
    tunable.tuning_enable(False)
    tunable.enable(False)
    if args.tuning_file:
        tunable.set_filename(str(args.tuning_file.resolve()))
        assert tunable.read_file(str(args.tuning_file.resolve())), 'TunableOp rejected the tuning file'
        assert tunable.get_results(), 'Tuning file contained no validated solutions'
    assert not tunable.tuning_is_enabled() and not tunable.is_enabled()
    meta_bytes = (args.pack_dir/'meta.json').read_bytes()
    meta = json.loads(meta_bytes)
    ctx = max(args.contexts) + max(args.tokens, 256)
    report = dict(torch=torch.__version__, hip=torch.version.hip,
                  device=torch.cuda.get_device_name(), pack=str(args.pack_dir),
                  pack_sha256=hashlib.sha256(meta_bytes).hexdigest(),
                  commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
                  diff_sha256=hashlib.sha256(subprocess.check_output(['git','diff','HEAD'])).hexdigest(),
                  source_sha256={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
                      ('src/drinkme/serving/deltanet_conv.py','src/drinkme/serving/kernel_route.py')},
                  free_before=torch.cuda.mem_get_info(), ctx=ctx,
                  candidate_blas=args.candidate_blas,
                  tuning_file=str(args.tuning_file) if args.tuning_file else None,
                  tuning_sha256=hashlib.sha256(args.tuning_file.read_bytes()).hexdigest() if args.tuning_file else None,
                  reference_conv_fla=args.reference_conv_fla, cases=[], complete=False)
    def save():
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2)+'\n')
    save()
    engine = build_engine(meta['hfRepo'], meta.get('revision'), str(args.pack_dir), stock=False, ctx=ctx)
    assert str(engine.device).startswith('cuda')
    assert 'auto' not in args.modes or engine.mtp_head is not None
    modelings = {sys.modules[type(m).__module__] for m in deltanet_conv.deltanet_modules(engine.model)}
    routes = {m: (deltanet_conv.reference(m), getattr(m, deltanet_conv.NAME)) for m in modelings}
    assert all(a is not b for a,b in routes.values()), 'FLA route did not install'
    original_blas = torch.backends.cuda.preferred_blas_library()
    report['reference_blas'] = str(original_blas)
    server = start_server(engine, '127.0.0.1', 0)
    port = server.server_address[1]
    report['health'] = serve_drive.health(port)
    assert port != DEFAULT_PORT  # an OS-chosen port, never serve's default
    warmup_task = serve_drive.PROMPT
    task = args.prompt_file.read_text() if args.prompt_file else warmup_task

    def select(arm, mode):
        for m, (old, new) in routes.items():
            setattr(m, deltanet_conv.NAME, old if arm == 'reference' and not args.reference_conv_fla else new)
        tunable.enable(arm == 'candidate' and args.tuning_file is not None)
        assert not tunable.tuning_is_enabled()
        torch.backends.cuda.preferred_blas_library(
            args.candidate_blas if arm == 'candidate' and args.candidate_blas else original_blas)
        os.environ['DRINKME_SPEC'] = mode
        engine._spec_plan = None
        engine._mtp_depth = None
        engine.reset_prefix_cache()

    try:
        for mode in args.modes:
            for arm in ('reference', 'candidate'):
                select(arm, mode)
                serve_drive.PROMPT = exact_prompt(engine, 128, warmup_task, 8)
                serve_drive.request(port, 8)
        for target in args.contexts:
            serve_drive.PROMPT = exact_prompt(engine, target, task, args.tokens)
            for mode in args.modes:
                case = dict(context=target, mode=mode, rows={'reference':[], 'candidate':[]},
                            prompt_sha256=hashlib.sha256(serve_drive.PROMPT.encode()).hexdigest())
                report['cases'].append(case)
                runs = args.long_runs if target >= 32768 else args.runs
                for i in range(runs):
                    order = ('reference', 'candidate') if i % 2 == 0 else ('candidate', 'reference')
                    for arm in order:
                        select(arm, mode)
                        torch.cuda.reset_peak_memory_stats()
                        before = serve_drive.metrics(port)
                        row = serve_drive.request(port, args.tokens)
                        after = serve_drive.metrics(port)
                        assert row['completion_tokens'] == args.tokens and row['prompt_tokens'] == target
                        row['mtp'] = {k:after[k]-before.get(k,0) for k in after if k.startswith('drinkme_mtp_')}
                        proposed = row['mtp'].get('drinkme_spec_proposed_tokens_total',0)
                        assert (proposed > 0) == (mode == 'auto')
                        row['peak_gib'] = torch.cuda.max_memory_allocated()/2**30
                        row['text_sha256'] = hashlib.sha256(row['text'].encode()).hexdigest()
                        case['rows'][arm].append(row)
                        print(f'{target=} {mode=} {i+1}/{runs} {arm}: '
                              f'TTFT {row["ttft_s"]:.3f}s, decode {row["decode_tok_s"]:.3f} tok/s', flush=True)
                        save()
                case['median_ttft_s'] = {a:statistics.median(r['ttft_s'] for r in rows) for a,rows in case['rows'].items()}
                case['median_decode_tok_s'] = {a:statistics.median(r['decode_tok_s'] for r in rows) for a,rows in case['rows'].items()}
                case['text_identical'] = len({r['text_sha256'] for rows in case['rows'].values() for r in rows}) == 1
                save()
    finally:
        server.shutdown(); server.server_close()
        for m, (_, new) in routes.items(): setattr(m, deltanet_conv.NAME, new)
        torch.backends.cuda.preferred_blas_library(original_blas)
        tunable.enable(False)
    assert (args.pack_dir/'meta.json').read_bytes() == meta_bytes
    report['complete'] = True
    save()
    print('VERDICT: PASS cold prefill HTTP A/B complete; transcript agreement recorded separately', flush=True)


if __name__ == '__main__':
    status = 0
    try:
        main()
    except SystemExit as exc:
        status = exc.code if isinstance(exc.code,int) else 1
    except BaseException:
        traceback.print_exc(); status = 1
    sys.stdout.flush(); sys.stderr.flush(); os._exit(status)
