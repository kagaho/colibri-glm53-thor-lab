# References

Primary sources checked November 24–25, 2026. Published measurements below
describe their own hardware, model formats, versions, and cache conditions.
They are context, not predicted Thor results.

- [Colibri v2.2.0 release](https://github.com/JustVugg/colibri/releases/tag/v2.2.0):
  current GLM Flash CUDA support and release-machine measurements.
- [Pinned source](https://github.com/JustVugg/colibri/tree/e479104ae01bdc40ce81e675299b1d230891b321):
  exact engine, downloader, benchmark, and CUDA-check implementation used here.
- [Native model](https://huggingface.co/Justvugg/GLM-5.3-Flash-colibri-int4-g64/tree/fe1d408f252c9c53a0d688aa54e41d9bdfdfd510):
  immutable converted checkpoint and its inherited model license.
- [Original model](https://huggingface.co/zai-org/GLM-5.3-Flash): model architecture,
  official reasoning modes, chat-template guidance, and attribution.
- [GLM Flash engine notes](https://github.com/JustVugg/colibri/blob/v2.2.0/docs/glm53-flash.md).
- [Benchmark protocol](https://github.com/JustVugg/colibri/blob/v2.2.0/docs/benchmarking.md).
- [CUDA implementation notes](https://github.com/JustVugg/colibri/blob/v2.2.0/docs/cuda.md).
- [Environment controls](https://github.com/JustVugg/colibri/blob/v2.2.0/docs/ENVIRONMENT.md).
- [Historical Spark report](https://github.com/JustVugg/colibri/discussions/1567):
  older v1.11.0 behavior; the missing-CUDA finding is not a claim about v2.2.0.
- [First-person SSD inference experiment](https://gist.github.com/matbanik/dcc8bb64cd77fe9c993e2f9555d5fd8e):
  different formats and engine versions, useful as a diagnostic example rather
  than a controlled comparison applicable to this machine.
- [First-person resident GGUF Spark measurement](https://dev.classmethod.jp/en/articles/dgx-spark-glm-5-3-flash-first-touch/):
  a separate in-memory experiment, not evidence for this native streaming run.

The original local PDF and private machine audit are excluded from the public
repository. The lab notebook records the decisions they informed.
