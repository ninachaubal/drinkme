"""drinkme — lossless BF16 weight compression, served.

serve / pack / bench / publish.

The bottle labeled DRINK ME: Alice "shut up like a telescope" — collapsed
smaller with every segment still there. That is the product in one image, and
the per-tensor round trip at pack time (codec/radix_pack: the encoder decodes
back and compares bit for bit to the source, or the pack is never written) is
the assert that keeps it true. (She also checked the bottle for poison
markings first; `drinkme check` is the same posture.)

The bf16 codec is radix (codec/radix.py: exponent tiers, literal
sign/mantissa, independently decodable 1024-weight blocks), in this repo's
pack container (codec/pack.py, docs/pack-format.md).
"""

__version__ = "1.1.0"

# The one lexicon. The suite IS the bench code at __version__: numbers
# compare within a semver major, and an existing metric name never changes
# meaning within one — new metric names may appear in minors.
MEASUREMENT = "wtf.petrichor.drinkme.measurement"
