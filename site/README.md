# Results site

The landing page at drinkme.petrichor.wtf is a static HTML file with no build
step. It uses vanilla JavaScript, inline SVG, a decorative canvas, self-hosted
fonts, and vendored Pretext. Assets and chart data are served from this directory.

The page introduces memory savings, model capacity, hardware speed, installation,
client setup, and atproto benchmark sharing. Its numbers are the measurement
records published on atproto (the `wtf.petrichor.drinkme.measurement`
lexicon). The committed `data/points.json` is a snapshot of such records, the
ones measured with drinkme 1.0.0 on 2026-09-27 ([chart data](#chart-data)). [The
aggregator](aggregator/README.md) writes the served `data/points.json` from
every record published on the network.

Charts accept BF16 results only, matching this release's supported checkpoint
format. Apple silicon results are charted too: Qwen3-1.7B and Qwen3-4B on an M4
([Metal](../docs/metal.md)).

## Preview

From the repository root:

```sh
python3 -m http.server 8765 --directory site
```

Open `http://localhost:8765/`. HTTP is required for data fetches and module
imports. This preview needs no Python environment changes or model-server restart.

## Design and assets

[DESIGN.md](DESIGN.md) describes the visual direction; [PRODUCT.md](PRODUCT.md)
defines the audience and messaging. Exact styling and behavior live in
[index.html](index.html).

Alice's opening shrink reflows text through Pretext. The illustration and text
control toggle her size. In the install section, drawn smoke rings continue
the Caterpillar's plate smoke up to the section's top. They're redrawn about
20 times a second while the section is on screen, and not at all off it.
Reduced motion disables automatic shrink and ribbon animation, and stands the
rings still. On phones, the Rabbit's opening paragraph follows his silhouette.
Semantic headings and text remain available without JavaScript.

Tenniel illustrations are public domain; image provenance is in
`art/*.webp.json`. Silhouette extents in `art/alice-bands.json` and
`art/rabbit-bands.json` are inlined for text flow; `art/make-cutout.py --bands
<cutout>` prints one (`art/caterpillar-bands.json` is the Caterpillar's, not inlined). Font licensing is in
`fonts/NOTICE` and `fonts/OFL.txt`; Pretext's license and notice are in
`lib/pretext/`. `art/make-cutout.py` documents the hero's cutout production.
`art/make-caterpillar.py` makes the Caterpillar in the install section: Commons
has no transparent version of that plate, so the script turns its paper to
alpha, and it measures the smoke that the page's drawn rings continue.
`art/make-og.html` generates the social preview, and `art/og-card.png.json` records
that image's provenance. The brand marks (Tangled, GitHub, Hugging Face) are
their owners' own files, used as shipped; the `.json` beside each in `art/` records its
source and the terms found for it.

## Chart data

The snapshot, `data/points.json`, holds [measurement
records](../lexicons/README.md) in the shape `drinkme bench` writes and
`drinkme publish` sends. `data/build_points.py` builds it from record files in
`measurements/`; explicit files select only those records:

```sh
python3 site/data/build_points.py                      # measurements/*.json
python3 site/data/build_points.py measurements/a.json  # only the files named
```

RFC-6901 pointers in `site.at` identify each number's source field; the one
derived number, `site.allocatableGB`, is `environment.memoryBytes` in decimal
GB. `data/check_points.py` resolves the pointers to detect drift; source
records are required for that check, but not for previewing the site. The
committed `points.json` holds 28 records measured with drinkme 1.0.0 on
2026-09-27 (UTC). Twenty are from one Strix Halo: sip and gulp for ten models,
Qwen2.5-72B's two among them, whose uncompressed arm doesn't fit. Eight are
sip on NVIDIA cards on Modal: Qwen3.8-27B on an A100 80GB, an H100 and an
L40S, where its uncompressed arm doesn't fit; Qwen3-8B on an L4, an A100 80GB
and an H100; gemma-4-31B-it on an A100 80GB and an H100. Modal's A100 80GB
pool gave PCIe and SXM4 cards, so those records carry two `deviceClass`
strings. The page shows them as one machine, "NVIDIA A100 80GB":
`MACHINE_GROUPS` in the page merges `deviceClass` values for the Machine
select, the table and the link only. Cells, points and labels keep each
record's own `deviceClass`, so the two variants are never medianed together. `measurements/` is not committed, so the check runs where those
record files are.

The build accepts records of this checkout's drinkme major version
([versioning](../docs/versioning.md#measurement-compatibility)). It
validates each record's identity fields and lists rejected records in
`excluded` and on stdout.

The speed chart groups records by device, `hfRepo`, resolved `revision`,
the compression profile (`compression.profile`), `environment.platform`, and version compatibility.
Each group has a separate label, median ratio, and sample count.
Each table row and each point's tooltip gives the group's contributor count
("1 contributor", "3 contributors"): the distinct `site.contributor` values the
aggregator writes. A record that has none counts as its own contributor.
The GPU select and the sip / gulp toggle choose what the chart highlights. The
select opens on All GPUs: every GPU's results in the chosen profile are in plum,
the other profiles' are grey, and there is no table (one row per model per GPU
would run too long). Below All GPUs, the GPUs are listed by most contributors; a
tie goes to Strix Halo, and after that to the GPU with the most records. Choosing
one GPU puts its points in plum, every other result in grey, and shows its
table. The choice is kept in the URL query as `?gpu=<slug>&profile=<profile>#speed`
(no `gpu` means All GPUs), where the slug is the `deviceClass` in lower case with
hyphens, so a shared link opens on the speed section. The hash stays free for the
section anchors `#intro`, `#fit`, `#speed`, `#start` and `#measure`. Both axes fit the data, and y always includes 1×.
The table under the chart has one row per model for that GPU and profile:
compressed and uncompressed tok/s and the speed ratio. Sizes are in the fit
table.
Uncompressed is the stock arm, `stock_decode_tok_s`: the same model's BF16
weights through drinkme's own runtime, the same kernel routes as the
compressed arm and no compression ([bench arms](../docs/bench.md#torch-benchmark-scope)), so
the ratio is compressed ÷ stock. The key under the chart is its legend
alone; what the baseline is and how far repeated runs move are in
[bench](../docs/bench.md#torch-benchmark-scope), not on the landing page.
Records also carry bench's other arms as data; the page shows only these two.
A record whose uncompressed arm didn't load gets a row, but not a point.
Where a row's records carry speculative decoding metrics, the row has second
lines, read from those records, "up to" the best prompt: under compressed,
`up to <max> tok/s` over `compressed_spec_decode_tok_s_{agent,chat,code}`;
under uncompressed, the same over `stock_spec_decode_tok_s_*`. There is no
speculation line under speed: the speed column already compares the arms, and
on cards that aren't bandwidth-bound compressed ÷ stock stays under 1× with
speculation too, while the tok/s lines show what speculation adds. Each
prompt's number is the median across the row's records, the same median as the
row's other numbers. The words "with speculative decoding" show once, on the
compressed line (the uncompressed one when there is no compressed line), and
are visually hidden on the other. Only a checkpoint with an MTP head, run under
torch, carries them (today Qwen3.8-27B), any profile; a row whose uncompressed
weights didn't load shows the compressed line only. A row without the metrics
shows nothing extra. The page computes the lines in `specLines` in its
grouping block, tested in `tests/test_site_points.py`. They never go in the
chart, which stays speculation off. Until records with these metrics are
published, the live table shows no speculation lines.

The fit section's table needs no machine, and no `points.json`: a weight
footprint belongs to the model, so its rows are written in the page as
`FIT_ROWS`. Each row gives the weights' size uncompressed and with sip and
gulp, in GB, largest first; the page computes how much smaller each profile
is. The numbers are the snapshot's records' `stock_weights_gb` and
`compressed_weights_gb` footprints (Strix Halo, 2026-09-27, drinkme 1.0.0); a
vision model's include its vision tower. A model too big to load uncompressed
has no `stock_weights_gb`, so its BF16 size is its checkpoint's safetensors
bytes, and the comment above `FIT_ROWS` says so.
The comment above `FIT_ROWS` says how to add a model. A profile with no record
shows a dash.

Capacity points have a `stock.outcome` other than `measured`.
`skipped_predicted_nonfit` includes the `budgetBytes` used for the decision;
`failed_load` includes the load error. The page has no chart of them: the
speed table lists each as a row whose uncompressed arm didn't load.

When no records match the current major version, the machine select and
the chart display an empty state explaining that no results are available.

## Checks for a site change

Check desktop and phone widths (1440px and 390px), reduced motion, and JavaScript
disabled. Verify text flow, command wrapping, navigation, chart loading, the
machine select and profile toggle, and the table's stacked cards on phones. Test
keyboard Tab through the table rows (each labels its point), Escape, and focus
restoration.

Hovering or tapping a point shows a one-line tooltip and highlights its table row.
Neither the tooltip nor a highlighted row moves the plot.

Check setup copy against [the README](../README.md), [hardware support](../docs/hardware.md) and [ROCm](../docs/rocm.md).
The page's commands match the README's exactly: `uv sync` once in a new
checkout, then `uv run --no-sync` for every command.
