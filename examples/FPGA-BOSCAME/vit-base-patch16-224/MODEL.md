# Model contract

The graph uses the HuggingFace ViT-B/16 ordering: patch convolution, flatten and
CLS/position additions; twelve pre-layernorm attention blocks; final layernorm,
CLS selection and a 768-to-1000 classifier.  The trace-friendly graph keeps the explicit QK scale
`1/sqrt(64)=0.125` after the dot product; the graph replacement folds that
MulOp into the QK Triton call, whose case uses `SCALE=0.125`. All dense
projection biases are explicit `bias_add` calls. The patch Triton kernel
accepts the convolution bias directly, so neither path drops the checkpoint
bias.

The Triton linear kernel reads checkpoint weights as contiguous `[N,K]` and
performs `A @ weight.T` internally. The graph replacement consumes the
original weight placeholder and removes its now-unused transpose. Attention
Q/K/V transposes remain in the Buddy graph and materialize contiguous head
tensors before the QK and PV calls. The numerical subgraph has 201 inputs:
the image followed by 200 captured parameter tensors. The main graph splits
one flat FP32 weight buffer into that same parameter order.

Every Triton case has fixed ranks, element type and grid in its `frontend.json`.
The graph replacement report must list the case and exact tensor shape for every
external call; missing shapes stay in `uncovered` and make the replacement step
fail when `--require-complete` is selected.
