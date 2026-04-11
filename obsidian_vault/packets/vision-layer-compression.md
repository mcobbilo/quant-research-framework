# Packet: Visual Token Overload Mitigation

- packet_id: optimization-vision-dpi-downscaling
- source: Phase 145 Context Reduction
- confidence: high
- status: promoted
- tags: doctrine:context_compression, geometry, latency

Claim
Visual processing latency and token bleed can be radically minimized without degrading analytical structure by aggressively dropping chart plotting resolution from 300 to 100 DPI.

Mechanism
Visual Large Language Models tokenize inputs dynamically by dividing the raster image into geometric tiles. By reducing the physical spatial dimension (`figratio 16:9, scale 1.0, DPI 100`), the quantity of tiles drops immensely ~60%, forcing the neural net to process structural shapes across fewer dimensions without losing mathematical relationships (e.g., higher highs).

Boundary
If the pattern requires reading highly granular sub-pixel text labels, downscaling text completely destroys alphanumeric clarity, making DPI compression dangerous for highly-textual screenshots.

Contradiction
More pixels equals deeper understanding. This holds generally true for real-world photo generation, but classical geometric chart topography thrives on low-res contour matching.
