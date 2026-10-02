# Design direction

An Alice in Wonderland storybook with psychedelic endpapers: Tenniel illustrations, Old Standard book type, and lilac, plum, pink, and acid green. Preserve generous reading space within that playful palette.

- Let prose flow around the illustrations’ actual silhouettes using Pretext. Keep headings readable on small screens and commands easy to scan and copy.
- Alice’s opening shrink should reflow the text around her. Keep decorative motion brief, preserve manual control, and respect reduced-motion preferences.
- The one lasting motion is the Caterpillar’s smoke in the install section: drawn rings continue his plate’s own smoke, in its line and ink, and rise slowly behind the words. It stops off-screen and stands still under reduced motion.
- Use natural scrolling, with an easy path from the introduction through hardware results to installation and benchmark sharing.
- The fit section's table gives every model's size uncompressed and with each profile, so a reader can check fit without choosing a machine. Speed comes after it.
- The results chart leads with one machine: its points in plum, every other result a recessive grey, read against a labelled 1× line. A table under it carries the numbers. Hover, focus, and tap must not move the plot. Label only the highlighted point, and never put a label on every point.
- Preserve semantic text, keyboard access, visible focus, accessible icon labels, and readable contrast. Never encode meaning in colour alone: pair it with text, a table, or a labelled baseline.

See [PRODUCT.md](PRODUCT.md) for audience and messaging. Keep exact colors, sizes, breakpoints, timings, and component behavior in [index.html](index.html), rather than duplicating them here.
