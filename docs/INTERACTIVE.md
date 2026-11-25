# Test the native model yourself

On Shockwave, run:

```bash
cd ~/Documents/Colibri
python3 scripts/chat.py --take-over
```

`--take-over` explicitly stops this lab's automated campaign and waits for its
server to exit before loading an interactive session. It does not stop any
Docker container. If you want the campaign to finish first, wait and run
`python3 scripts/chat.py` without the flag afterward.

The helper uses the native int4 checkpoint and the conservative CUDA setting
`COLI_CUDA_TC_GEMM=0`. It keeps explicit RAM and device-tier budgets, 8K context,
low reasoning effort, and a 256-token output cap including reasoning. The
initial model load takes time. After the chat prompt appears, try:

```text
What is 17 multiplied by 23? Answer with the integer only.
```

The expected answer is `391`. Then ask your own questions and inspect the token
and timing statistics printed after each answer. Exit with Ctrl+C.

To compare CPU behavior or permit longer answers:

```bash
python3 scripts/chat.py --cpu --max-tokens 512
```

Run one interactive session at a time. Interactive prompts are not written to
this repository's benchmark logs by this helper. An interrupted automated
campaign keeps its partial raw results. To run a fresh campaign after exiting
chat, use `python3 scripts/campaign.py`; this starts new attempts rather than
claiming a continuation is an independent trial.
