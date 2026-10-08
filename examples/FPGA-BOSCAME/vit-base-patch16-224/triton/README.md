# ViT W8A8 operator checks

The three quantized stages use explicit contiguous buffers:

```text
quantize:   X[f32,M,K] -> Q[i8,M,K] + row_scale[f32,M]
matmul_i8:  Q[i8,M,K] @ W[i8,N,K]^T -> Acc[i32,M,N] (accumulate)
dequantize: Acc[i32,M,N] * row_scale[M] * column_scale[N] -> Y[f32,M,N]
```

`B`/`W` is physically `[N,K]`, matching the ViT model weight serialization.
The signed quantizer uses `round(x / scale)` with ties away from zero,
saturates to `[-127,127]`, and assigns scale `1` to an all-zero row.  AME
matmul cases clear `C` in their launcher and the kernel retains `+=` semantics.

The reproducible host sweep below validates the patch projection and all linear
shapes used by the graph rewrite (196/197/1 rows and the 768/3072/1000 output
widths):

```bash
cd examples/FPGA-BOSCAME/vit-base-patch16-224/triton
python build.py --host --jobs=4 \
  --case quantize_196x768 --case quantize_197x3072 \
  --case quantize_197x768 --case quantize_1x768 --case quantize_768x768 \
  --case matmul_i8_196x768x768 --case matmul_i8_197x3072x768 \
  --case matmul_i8_197x768x3072 --case matmul_i8_197x768x768 \
  --case matmul_i8_1x1000x768 \
  --case dequantize_196x768 --case dequantize_197x3072 \
  --case dequantize_197x768 --case dequantize_1x1000
```

Each launcher prints `verify <case>: PASS ... max_abs_error...` into
`build/<case>/host/output.log` and prints a `cycles <case>:` line for timing;
the quantizer prints one timing line per fixture trial.
The NR build uses the same command without `--host`.  `check_nr_elf.py`
reports 21 AME instructions for each integer matmul and zero AME instructions
for the scalar quantize/dequantize stages.

The `patch_embed_196x768` adapter keeps its four-tensor FP32 graph ABI while
running the same W8A8 sequence internally. It materializes im2col, quantizes
the 196 activation rows and flattened 768 weight rows, invokes the integer
projection, then dequantizes and adds the FP32 bias. Its independent host
oracle allows quantization error up to `3e-3`; the deterministic fixture passes
with maximum absolute error below `6e-5`.
