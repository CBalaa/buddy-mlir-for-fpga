# Text / tokenizer path

EfficientNet-B3 is an **image classifier**. There is no vocabulary, BPE,
chat template, or on-board text encoder.

Qwen's `model/text/` tree (tokenizer resource, unicode tables, encode/decode)
is **not built** for this example. Related tools under `model/tools/` exist as
stubs that exit with an explicit N/A message so deployment scripts can keep
path parity with the LLM example.

Board input is a fixed **NCHW** tensor (`host_reference.py --board-input`), and
UART verification prints **argmax / logit bits**, not generated text.
