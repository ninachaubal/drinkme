# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

People running local models who want lower memory use or a larger model on their existing hardware. They need a clear path to installation and use.

## Product Purpose

drinkme runs local models with smaller weight storage while preserving checkpoint weights. The site should help visitors understand the benefit, install drinkme, and connect their existing client.

## Messaging

- Lead with less memory and room for larger models; generation can get faster, too.
- Memory savings depend mostly on the model and weight format. Speed also depends on hardware.
- ROCm covers discrete AMD GPUs as well as Strix Halo; CUDA covers NVIDIA. Automatic dependency setup has narrower coverage than runtime support.
- Apple silicon: BF16 kernels passed bit-exact checks on an M4 and reached 81–91% of measured memory bandwidth. Whole-model BF16 serving is not yet tested on a Mac.
- Hardware results help visitors estimate what they might get. Compare drinkme with stock at the same weight precision, and name the baseline runtime.
- Benchmark publishing to atproto lets people share measurements under their own identity and build independent views. The site shows the measurement records published on atproto.
- Use plain, inviting copy. Keep facts and labels accurate without making declarations of honesty the brand.

See [DESIGN.md](DESIGN.md) for visual direction. Supported commands and APIs belong in the project’s operator documentation; measurements come from the published records, through the site’s data files, not its copy.
