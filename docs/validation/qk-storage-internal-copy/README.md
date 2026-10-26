# QK storage normalization and internal-copy evidence

This checkpoint records real artifacts from
`attention_qk_12x197x197x64`, its storage normalization and the independent
internal-copy implementation. No full IR, object files or machine-specific
paths are copied into this directory. Artifact and relevant source SHA256
values are in [artifact-source-hashes.json](artifact-source-hashes.json);
the final emitted module equals the input of the completed full-grid simulator.

## Actual frontend excerpts

The following excerpts retain the real SSA names and operation order from
the first complete branch of `before.frontend.mlir` and
`after.frontend.mlir`. Module/function wrappers, address calculations, input
copies, shape expand/collapse operations, other branches and large
`contract.*` dictionaries are omitted. `#map` is the two-dimensional identity
map. These are excerpts for review, not standalone test modules.

Before normalization, bufferization preserves the zero tensor for the later
add and therefore initializes a distinct matmul product buffer:

```mlir
%cst = arith.constant 0.000000e+00 : f32
%cst_0 = arith.constant 1.250000e-01 : f32
%alloc = memref.alloc() alignment = 64 : memref<16x16xf32>
%alloc_1 = memref.alloc() alignment = 64 : memref<16x16xf32>
linalg.fill ins(%cst_0 : f32) outs(%alloc_1 : memref<16x16xf32>)
linalg.fill ins(%cst : f32) outs(%alloc : memref<16x16xf32>)
// Actual A/B allocation and input-copy operations occur here.
%alloc_5 = memref.alloc() alignment = 64 : memref<16x16xf32>
memref.copy %alloc, %alloc_5 : memref<16x16xf32> to memref<16x16xf32>
linalg.matmul
  ins(%alloc_2, %alloc_4 : memref<16x64xf32>, memref<64x16xf32>)
  outs(%alloc_5 : memref<16x16xf32>)
linalg.generic {indexing_maps = [#map, #map, #map],
                iterator_types = ["parallel", "parallel"]}
  ins(%alloc, %alloc_5 : memref<16x16xf32>, memref<16x16xf32>)
  outs(%alloc : memref<16x16xf32>) {
^bb0(%in: f32, %in_7: f32, %out: f32):
  %16 = arith.addf %in, %in_7 : f32
  linalg.yield %16 : f32
}
// Actual expand/collapse shape operations occur here.
linalg.generic {indexing_maps = [#map, #map, #map],
                iterator_types = ["parallel", "parallel"]}
  ins(%alloc, %alloc_1 : memref<16x16xf32>, memref<16x16xf32>)
  outs(%alloc : memref<16x16xf32>) {
^bb0(%in: f32, %in_7: f32, %out: f32):
  %16 = arith.mulf %in, %in_7 : f32
  linalg.yield %16 : f32
}
// Exact valid-range external output copy follows.
```

After normalization, the existing add captures the exact scalar zero and
uses the matmul result as its destination; the extra product allocation and
initialization copy disappear:

```mlir
%cst = arith.constant 0.000000e+00 : f32
%cst_0 = arith.constant 1.250000e-01 : f32
%alloc = memref.alloc() alignment = 64 : memref<16x16xf32>
%alloc_1 = memref.alloc() alignment = 64 : memref<16x16xf32>
linalg.fill ins(%cst_0 : f32) outs(%alloc_1 : memref<16x16xf32>)
linalg.fill ins(%cst : f32) outs(%alloc : memref<16x16xf32>)
// Actual A/B allocation and input-copy operations occur here.
linalg.matmul
  ins(%alloc_2, %alloc_4 : memref<16x64xf32>, memref<64x16xf32>)
  outs(%alloc : memref<16x16xf32>)
linalg.generic {indexing_maps = [#map, #map],
                iterator_types = ["parallel", "parallel"]}
  ins(%alloc : memref<16x16xf32>) outs(%alloc : memref<16x16xf32>) {
^bb0(%in: f32, %out: f32):
  %16 = arith.addf %in, %cst : f32
  linalg.yield %16 : f32
}
// Actual expand/collapse shape operations occur here.
linalg.generic {indexing_maps = [#map, #map, #map],
                iterator_types = ["parallel", "parallel"]}
  ins(%alloc, %alloc_1 : memref<16x16xf32>, memref<16x16xf32>)
  outs(%alloc : memref<16x16xf32>) {
^bb0(%in: f32, %in_6: f32, %out: f32):
  %16 = arith.mulf %in, %in_6 : f32
  linalg.yield %16 : f32
}
// Exact valid-range external output copy follows.
```

The normalization helper maps the original add's operands without swapping
them and clones its attributes. Existing canonicalization subsequently prints
the scalar constant on the right; the two excerpts show that actual change.
The computation stages remain ordered matmul → independent `+0` → scale
multiplication → valid-range store. There is no new FMA, reassociation or
fastmath flag, and `MatmulConverter::skipC` is unchanged. This evidence does
not assert that arithmetic NaN payloads remain identical.

The actual planned add records its constant capture explicitly (unrelated
binding fields omitted):

```mlir
plan.vector = {engine = "rvv", vector_bytes = 64 : i64,
  elements = 256 : i64, tail = "none", fp_policy = "preserve",
  scalar_operands = [{kind = "constant-broadcast", operand = 1 : i64,
                      value = 0.000000e+00 : f32}]}
```

Emission creates the exact scalar and its vector broadcast inside the
pointwise loop body, before the cloned arithmetic operation. The scale
still comes from its separate `0.125` fill, so its multiplication remains
after the add rather than being folded into the matmul.

## Recorded results and limits

- [storage-change-report.json](storage-change-report.json): four branches,
  alloc 20→16, copy 16→12, private initialization copy 4→0, task 40→36;
  matmul/addf each remain four; bank high-water bytes are 5120/5120.
- [old-new-branch-report.json](old-new-branch-report.json): old bufferized
  frontend versus normalized frontend at full/full, full/tail, tail/full and
  tail/tail coordinates, all in head zero. The four programs write
  256+80+80+25=441 distinct C elements. The original script compares the entire
  12×197×197 C array, so `compared_elements=465708` includes 465267 untouched
  elements with their deterministic initial C values. Of the complete array,
  465676 non-NaN elements match bit for bit and 32 NaN elements match
  classification; within the written region that is 409 non-NaN and 32 NaN
  elements. Inputs, DDR guards and exact access rectangles pass.
- [old-emitted-branch-report.json](old-emitted-branch-report.json): the same
  old frontend versus the emitted target path, with the same comparison and
  access criteria. Arithmetic NaN classification and raw-copy payload
  preservation are separate claims.
- [qk-final-build-report.json](qk-final-build-report.json): final compiler
  frontend through target object assembly passes, with all generated artifact
  SHA256 values. `artifact_dir` is a relative reproduction label; large
  artifacts are not bundled here.
- [existing-examples-summary.json](existing-examples-summary.json): all three
  original examples pass build/check/simcheck and linked-ELF audit. Stack
  bounds are 656/1216/1168 B under the unchanged 65536-B capacity.
- [verification-checkpoint.json](verification-checkpoint.json): repository
  and design checks at this checkpoint, including the completed full-grid QK
  simulation. No full-model result or board run is claimed.
- [qk-full-simcheck-report.json](qk-full-simcheck-report.json): one normal input
  runs all 2028 QK programs in reverse grid order and checks all 465708 output
  elements against the independent ordered-f32 oracle. Every output matches
  bit for bit, with maximum absolute error zero. The module and kernel-source
  hashes match the final compilation evidence. This full-grid normal-input
  result is separate from the four-coordinate special-value comparison above;
  it does not extend special-value coverage to every program. No board or
  hardware performance conclusion is drawn from either simulation.
- [final-suite-summary.json](final-suite-summary.json): a fresh compilation of
  all 34 ViT cases requests target assembly; 16 reach assembly and 18 stop in
  the frontend. Those 18 cases represent seven behavior groups: PV masked
  access/valid-shape handling, dequantization indexed scale reads, GELU
  transcendental semantics, softmax exponentiation and reductions, linear
  loop content dependencies, patch im2col memory access, and quantization
  scalar store/reduction/rounding/conversion. First diagnostics remain
  separate from later requirements. The suite's expected nonzero exit denotes
  these blocked cases rather than 34 successful compilations.

The independent internal-copy recipe preserves genuine copies between
distinct private SPM roots. Its plan includes complete physical endpoint
envelopes, window/bank capacities, nonoverlap, exact tile coverage and final
completion. Same-bank nonoverlapping roots are permitted; same-root aliases
are rejected. Each tile temporarily uses fenced ACC0 without new staging,
arbitrary physical TR/ACC conversion or cross-operation register residence.
`PlanInternalCopies.py` separately tests raw bytes, including NaN payloads,
poisoned SPM, rereading the source, guards and rejection gates. QK's eliminated
initialization copies do not substitute for those tests. See the
[complete validation record](../../BOSCAMEKernelContractValidation.md#qk-storage-normalization-and-internal-copies).
