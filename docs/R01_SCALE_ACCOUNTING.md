# R01 scale accounting for 20M → 100M

Status: `PLANNING_ONLY`. Issue: #713. No training or paid compute is authorized here.

The mechanical MODEL-341 control has exactly 20,613,440 parameters with a byte vocabulary of 256.
That number is useful for identity, but it is not by itself a stable compute or capacity axis once the
project fits a learned tokenizer. At fixed transformer geometry, a larger vocabulary adds embedding
parameters and output-projection compute. At fixed total parameters, compensating for those embeddings
would shrink the transformer. Either comparison can silently confound the 20M/50M/100M sweep.

`src/twelve_six/scale_accounting.py` therefore exposes three separate quantities:

- exact parameter breakdown for the repository's GQA + SwiGLU decoder;
- dominant matrix-multiplication FLOPs/token, split into transformer projections/MLP, attention context,
  and vocabulary projection;
- fixed-geometry vocabulary sensitivity.

The FLOP number is a planning estimate, not measured runtime. It excludes norms, activations, softmax,
optimizer work, communication, allocator effects, and kernel overhead. Hardware decisions must use
measured throughput and peak memory from bounded pilots.

The checked-in 50M and 100M geometries are arithmetic probes only. They are deliberately not ModelSpec
authority. D01 may freeze a future 100M ModelSpec only after learned-20M evidence and the R01 matched
sweep justify it.

Research rationale: DeepSeek LLM shows why simple parameter-count compute proxies can be materially
inaccurate at small scale; vocabulary-scaling work shows vocabulary size is itself a scaling variable;
MobileLLM supports deep-thin/GQA as a useful sub-billion candidate bias. These sources motivate
measurement, not automatic architecture adoption.
