## Lessons

- Unstructured LLM responses generating trades fail unpredictably -- bind LLM text outputs to Pydantic JSON schemas to mathematically constrain string generation (Phase 141).
- Python stateless scripts looping large transactions run out of memory or crash mid-transaction during internet outages -- wrap critical logic in Temporal Durable Workflows so the background state recovers automatically upon rebooting (Phase 142).
- Passing 300 DPI matplotlib images to Gemini Vision burns maximum token tiling and drastically inflates latency (25s+) -- compress the geometry to 100 DPI and `figscale=1.0` to cut latency and tokens by ~60% without losing geometric context (Phase 145).
- Print logs passing "Security Warnings" directly bypass block checks allowing execution -- enforce system logic securely by using python Exception bounds natively (Fail-Closed state) (Phase 139).
