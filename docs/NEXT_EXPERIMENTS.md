# Follow-up experiments

Complete the initial load, correctness, and two-profile smoke study before
expanding the matrix. Record negative results and any changed settings.

1. **Repeat and alternate profile order.** Use at least three independent server
   starts per profile, alternating CPU/CUDA and CUDA/CPU. Report medians and
   spread; identify warm OS caches separately from warm expert caches.
2. **Sweep expert residency.** Hold the RAM expert cache at 32 GB while testing
   8, 16, and 32 GiB GPU tiers, one at a time. Then vary the RAM cache at a fixed
   device tier. Keep physical headroom and measure allocation peaks on UMA.
3. **Prefill versus decode.** Add fixed tokenizer-counted prompts at roughly
   128, 1K, 4K, and 8K tokens and a sufficiently long completion. Use actual
   engine timing boundaries before reporting a decode-specific throughput.
4. **Storage policy.** Compare buffered and supported direct-I/O paths only
   after checking the GLM engine's actual controls. Record alignment, request
   size, queueing, OS-cache state, bytes read, and an independent read control.
   Do not substitute a drive's sequential rating for expert-read performance.
5. **Prefix caching.** Separately enable the content-addressed store, then test
   repeated system/tool prefixes, follow-up questions, and intervening side
   requests. Compare reused tokens, first output, memory, and answer correctness.
6. **Runtime dense precision.** Compare 4 versus 8 bits with the native int4
   routed experts unchanged. Rebudget RAM/device copies before loading.
7. **Quality and useful work.** Add a versioned set of coding and reasoning
   tasks with objective checks. Keep tests separate from performance prompts,
   avoid unbounded generated-code execution, and disclose sample selection.
8. **Steady-state power and thermals.** Record the unchanged operating point,
   temperature, clocks, and watts using supported meters. Distinguish estimated
   energy from measured energy; do not change power modes inside a comparison.
9. **Optional other-engine comparison.** A GGUF comparison changes format and
   often quantization. Label it as a practical alternative, not a controlled
   native-container engine comparison: llama.cpp does not read this container.
