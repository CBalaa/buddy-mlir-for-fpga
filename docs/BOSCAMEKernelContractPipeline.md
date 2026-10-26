# BOSCAME kernel contract, planning and emission

The current pipeline separates the frontend's computation and memory semantics
from BOSCAME placement and access routes. The frontend emits standard
Linalg/MemRef/SCF with `contract.*` attributes. `--bosc-ame-plan` rewrites that
buffer graph and records concrete `plan.*` decisions. `--bosc-ame-emit` validates
the entire plan and mechanically expands it into BOSCAME and vector operations.

The boundary is independent of Triton. Triton conversion currently lives in
this repository for development, but the contract does not require Triton or
BOSCAME dialect operations. Moving that frontend to another project must not
make it depend on the target planner. Other frontends can produce the same
standard IR and contract.

## Running the stages

For a Triton kernel, the frontend's opt-in contract mode includes bufferization:

```bash
buddy-opt kernel.ttir \
  --triton-to-linalg-experimental='contract=true caller-contract=args_aligned64_nonalias launch-grid=triton_matmul_i8_196x768x768:7,6,1' \
  -o kernel.contract.mlir
buddy-opt kernel.contract.mlir --kernel-contract-verify \
  --bosc-ame-plan='plan-report=true' --bosc-ame-plan-verify \
  -o kernel.planned.mlir
buddy-opt kernel.planned.mlir --bosc-ame-emit -o kernel.boscame.mlir
buddy-opt kernel.boscame.mlir --lower-bosc-ame -o kernel.llvm.mlir
buddy-translate kernel.llvm.mlir --buddy-to-llvmir -o kernel.ll
```

Use the kernel's actual symbol and launch grid. The grid is a caller promise
used to specialize full and tail programs; it does not authorize accessing
masked rows. `contract=false` remains the default frontend mode and produces
the ordinary target-independent lowering result.

Handwritten buffer-form Linalg can use `--kernel-contract-annotate` before
verification and planning. It must supply any ABI or program facts needed for
its accesses. Both compatibility names, `--linalg-to-boscame` and
`--lower-linalg-to-boscame`, use the same contract, planner and emitter. They
accept a neutral contract or an already planned module. Their limited input
adapter can also normalize supported handwritten buffer-form Linalg and
synchronous SPM/DMA inputs; it does not retain a second planner. Incompatible
legacy target metadata or geometry overrides produce migration diagnostics.
Explicit stages retain the intermediate plan for inspection.

The compatibility names expose the V1 recipe subset. Independent DDR-to-DDR
copies, external fills, SPM-to-SPM copies outside the explicit internal recipe,
and unproved broadcasts or unsupported scalar expression generics do not regain the retired
lowering fallback. Unsupported
inputs diagnose their constraint; standalone target DMA operations remain
available through `--lower-bosc-ame`.

There is no public global `store-policy` option, CLI alias, configuration field
or replacement policy switch. Remove old `bosc_ame.store_policy` attributes and
rerun planning. Old command-line options are rejected. The planner decides
each operand and boundary access separately; one matmul may use SPM A/C and
direct DDR B.

## Computation, content and storage

These are distinct objects:

| Object | Meaning |
| --- | --- |
| Standard operation | The computation and its actual memory reads/writes |
| Contract task and scope | Dependencies, completion boundary and modeled control flow |
| Content version | The contents produced by a writer, including inherited bytes of a partial write |
| Logical storage root and view | Allocation/argument identity, descriptor offset, sizes and strides |
| Physical window | A bank, byte offset and capacity selected for a private root |
| Operand/access route | The concrete DDR, SPM or host access used by an operation |

Signed i8 × i8 → i32 dot accumulation is normalized to `linalg.matmul` with the
original accumulator as its `outs` value. Supported zero-product-plus-integer-
add forms can be folded under private-use and alias proofs. Computation fusion
does not consult a storage policy and does not itself authorize external C
forwarding. Floating operations and flags retain their declared order; no new
FMA or reassociation is introduced by emission.

Before One-Shot bufferization and final graph annotation, rectangular memory
extents can also prove a loop's full prefix and one partial last iteration.
For static signed, positive-step loops with a proven nonwrapping IV domain,
the frontend clones the prefix loop and places the original last body after
it with a constant IV. Every original `iter_arg` in that last body uses the
corresponding prefix result, and the last body's yields replace the original
loop results. It preserves operation order, effects, origin provenance and
the padded compute shape. Divisible loops need no split; singleton short-K
domains already permit constant valid extents. Unknown bounds, unsigned
comparisons and unproved arithmetic retain their original accesses and
diagnostics.

Launch-axis and loop-axis proofs remain independent. For PV's K=197/BK=64,
the full loop runs blocks 0–2 and a fixed block 3 copies only five K lanes.
M's final program independently copies five rows. The private A/B tiles stay
16×64 and 64×16, initialized with the original `other` in excluded lanes;
the frontend never broadens a DDR read to cover padding. Dense short-K layout
checks use the statically proven valid rectangle rather than the padded
logical width. The final graph is rebuilt after these rewrites and
bufferization, including the prefix's init/backedge/result and the last
body's exit consumers. The existing floating path still computes each full
BK-sized product in order, then adds that product to the incoming accumulator.
ACC32 transport of its bits does not select floating AME arithmetic.

Known zero contents are normalized before contract graph construction. A copy
from private, completely initialized zero storage may become a local fill,
retaining the exact scalar bits, when aliases cannot change or escape before
any execution of that copy. A later write is safe only after the copy's entire
enclosing scope in the source allocation's block. A write later in the same
loop, including after an inner loop but before its next outer iteration,
still prevents rematerialization. Nonzero, partial or unproved contents keep
their copy and content dependencies. This does not relax the V19 control-flow
rule or label an invariant capture as a loop initializer.

For signed integer products, a separate plan proof can defer a complete private
i32 zero fill to its sole matmul consumer. `plan.fill` records
`engine="ame", kind="deferred-zero"` and the consumer task;
`plan.route.initialization` records the producer task and either `zero` or
`first-iteration-zero`. The verifier rechecks actual uses, full coverage, scope,
nonempty static loop and absence of intervening observers or writers. The
logical fill remains a content-producing task; emission preserves its completion
in an empty `scf.execute_region`, with no memory traffic. Its first consumer
materializes the value with `bosc_ame.tile.zero_acc`. Subsequent external K
iterations load and store the accumulated SPM contents normally. Each conditional
arm completes its own register lifetime after input DMA completion. A partial
padding fill, an observed initial value, or an incoming C snapshot cannot use
this recipe. Generic `C += A*B` kernels retain their C load.

The [zero accumulator target contract](BOSCAMEZeroAccumulator.md) specifies the
restricted same-register clear instruction, configuration and hardware evidence.
Model overwrite calls express zero initialization in their own neutral kernel
entry, rather than relying on a host memset or a call-to-call assumption.

Quantization uses ordinary scalar Linalg expressions: abs/maxnum reduction,
scale arithmetic, explicit scalar broadcasts, comparisons, saturation and
casts. Pointwise plans record a common lane count sized for the widest value;
narrow i8 results and i1 masks do not acquire wider memory footprints. Each
operand keeps its actual type and byte accounting. Payload operations and
floating multiplication order are cloned in order. Captured constants have
explicit per-operation broadcast decisions; dynamic scalars are explicit
operands with empty indexing maps. Singleton scale boundaries are exact copies
through private SPM, followed by a one-lane vector load/extract or fill; they
are not unsupported raw external scalar stores. Ordered-add and maxnum
reductions have distinct verified recipes.

A separate tensor storage normalization handles an existing floating add of a
zero-splat tensor and a single-use zero-initialized matmul result in the same
block. It replaces only that add's zero-tensor read with the exact scalar
constant, retaining its signed-zero bits and cloning the original operand
mapping and fastmath flags. Existing canonicalization can subsequently put
the constant on the right; the actual before/after excerpts are retained in
the [evidence bundle](validation/qk-storage-internal-copy/README.md).
The matmul result becomes the add's destination, allowing bufferization to
avoid preserving a separate tensor snapshot and its initialization copy. The
floating add remains after the matmul; this is not accumulator fusion or
elimination of an addition. `MatmulConverter::skipC` is unchanged. Nonzero,
multi-consumer or unproved cases retain their original SSA storage needs.

The contract records real reads and partial-write inheritance. Complete writes
kill old content while preserving required WAW ordering. A supported serial
loop binds its carried root to iter, backedge and exit content versions;
incomplete scopes cannot prove cross-boundary fusion or forwarding. Task IDs
are function-local, unique and topologically increasing; they need not be dense.

The planner binds actual operands, removes only proven redundant storage events
and rebuilds tasks, versions and dependencies. `plan.task_map` retains every
source task and records its surviving target or elimination. Its cardinality
is checked against the pre-rewrite `plan.source_task_count`. Emission preserves
these as `bosc_ame.task_map` and `bosc_ame.source_task_count`; eliminated tasks
are not reintroduced as executable events.

## Routes and current limits

Private C carried across K iterations retains private accumulation storage.
SPM capacity can force another proven direct candidate for A/B, but cannot
justify moving external writes or dropping snapshots. Full terminal snapshots
may be forwarded only after proving coverage, no other uses, alias safety,
execution order and all tile addresses. Candidate legality is checked before
traffic comparison; capacity retries retain the same safety requirements.

A terminal pointwise output can also write directly to a proven external
destination. The producer and its full final copy must be in the same block;
intervening observations, additional consumers, partial coverage or unknown
aliasing keep the copy. If the generic reads its old `outs` value, forwarding
passes that original private buffer as a separate input. Rebinding the output
must not replace its initial contents with those of external DDR. Address-only
views defined after the producer are cloned before it only when their SSA
operands can also be made available safely.

Boundary copies cover the effective source and destination views exactly and
use the same physical contiguous axis. The existing pure-DMA plan keeps
`engine="dma"`, `chunks={count,bytes,ddr_stride,spm_stride}` and
`wait="after-each-chunk"`. Each actual DMA start address and length must be
64-byte aligned. Multiple rows require legal addresses for every row; a
single row does not use its parent pitch, so that unused pitch need not be
64-byte aligned or at least the row length. Its type, units, nonnegative
value and integer range still require validation.

Boundary engine selection is automatic and deterministic: retain a wholly
legal DMA route; if pure DMA fails, try the bounded matrix-register route
below; if that candidate fails, try the existing segmented DMA/RVV route.
Each candidate proves its own endpoints, layouts and exact access range.
There is no global storage-policy switch or uncalibrated latency ranking.
A failed matrix candidate does not weaken DMA or simulator checks.

The matrix-transfer design uses an independent atomic
`bosc_ame.matrix.copy` operation for raw e32 ACC32 transport between DDR and
SPM in either direction. It performs no arithmetic or element conversion;
f32 bits, including NaN payloads, infinities and signed zero, are preserved.
Both endpoints are static rank-two views with the same physical contiguous
axis and matching shape and element type. Logical i8/i16/i32/i64/f32/f64
storage is eligible only when raw32 words cover complete elements and exact
row bytes; no additional instruction widths are enabled. Each effective pointer is
independently proven naturally aligned to 4 B; both byte pitches and the
valid row byte count are multiples of 4. The pitches remain independent.
Each row is accessed exactly, excluding pitch holes, padding and neighboring
outputs. This is neither packing nor transposition.

A tile uses one ACC32 and is bounded by 64 rows, 256 B per row and 16384 B.
Larger rectangles are split into separately verified tiles. These bounds
come from the existing verified 64-by-64 i32 compiler subset, not a CSR
capacity measurement or a newly measured board configuration. For example,
a `[64,16]` f32 source with strides `[197,1]` and a packed destination with
strides `[16,1]` has independent 788 B and 64 B pitches. Its full tile moves
4096 B; a five-column tail moves 20 B per row and must not widen to 64 B.
Reverse transfer swaps the actual endpoints, layouts, offsets and pitches.

The explicit matrix plan retains `chunks={count,bytes,ddr_stride,spm_stride}`
and records the following additional data:

```text
engine = "matrix"
source_layout = {shape = array<i64: ...>, strides = array<i64: ...>, element_bits = 32}
target_layout = {shape = array<i64: ...>, strides = array<i64: ...>, element_bits = 32}
tiles = [
  {id, predecessor, source_offset, target_offset, rows, row_bytes,
   source_pitch, target_pitch, instruction_bits = 32,
   register_class = "acc", register_count = 1, mtype = 65602}, ...]
temp_bytes = 0
total_bytes = chunks.count * chunks.bytes
completion = last_tile_id
wait = "after-each-tile"
reason = actual_matrix_selection_reason
```

Layout shape and strides use `DenseI64ArrayAttr`; layout strides are in
elements. Tile offsets and pitches are bytes relative to the original
endpoint's effective pointer, after applying its descriptor offset once.
The first predecessor is -1 and each later predecessor names the preceding
tile. Coverage is checked per row in both endpoints, with checked arithmetic;
matching total byte counts alone are insufficient. `total_bytes` excludes
pitch holes. `temp_bytes=0` means no new staging allocation, while the single
ACC32 occupancy is recorded separately. The emitter consumes the verified
tile list and configuration; it does not select a different engine.

No live TR/ACC value may span a matrix transfer. Target lowering keeps
the leading fence, configuration, `mlce32.m`, `msce32.m` and the final fence in one side-effecting
bundle with acc0 and memory clobbers. The bundle contains no runtime call
and exposes no register value across operations or calls. Subsequent
compute establishes its required tile/mtype configuration again. Logical
task dependencies, aliases, SPM access checks and memory ordering remain
required independently of the clobbers; completion is after the last whole
bundle. The fixed `mtype=65602` is the current NR recipe encoding. The NR
encoding version differs from the accompanying PDF, so its bit fields must
not be inferred directly from that PDF.

The route is implemented in the production planner, verifier, emitter and
target lowering. `PlanMatrixTransfers.py` validates the full pipeline, strict
raw-byte simulation and optimized RISC-V object generation. The earlier QK
input-only evidence and the subsequent complete QK compilation and branch
comparisons are recorded separately in the validation record.
There is no new board validation or support claim for arbitrarily unaligned
bases, stores of other instruction widths or CSR-based capacity discovery.
The closed logical-type set excludes f16/bf16; eligible i16 rectangles can
use raw32, while an i16 copy that needs RVV still fails explicitly.

Genuine internal copies use a separate explicit
`plan.transfer={role="internal", engine="matrix", recipe="spm-acc32-spm",
from="spm", to="spm", temporary_window="none",
ordering="synchronous-fenced-tiles", ...}`. Each endpoint records its operand,
bank, window offset, capacity and complete physical byte envelope `[begin,end)`,
alongside independent layouts, `chunks={count,bytes,source_stride,target_stride}`
and the verified tile list. `temp_bytes=0`, total valid bytes, final completion
and `wait="after-each-tile"` are explicit. The selection reason is
`nonoverlapping-spm-exact-acc32-rectangle`. No DDR endpoint or DMA channel is
invented for this route.

Both roots must already have legal SPM placements and be distinct private
roots. Their full rectangular envelopes, including pitch holes, must fit
their declared windows and banks; same-bank envelopes must not overlap. Even
disjoint views of one root are rejected. The same rank-two, common-axis,
closed-type and exact four-byte alignment rules apply. Each tile uses a
transient fenced ACC0 load/store bundle, without a staging allocation,
cross-operation register residence or physical TR/ACC format conversion. The
planner records the complete decision, the verifier recomputes both endpoints
and coverage independently, and emission consumes that decision.

An explicit `engine="segmented"` plan can instead contain DMA and exact RVV
steps. `chunks.count` remains the logical row count and `chunks.bytes` the
valid bytes per row. `segments` is a flat ordered list containing `id`,
`predecessor` (-1 for the first step, otherwise the previous id),
`engine="dma"|"rvv"`, `source_offset`, `target_offset`, `bytes` and `reason`.
Offsets are bytes relative to the actual copy endpoint descriptors and
already include row strides; emission must not repeat the list by row. RVV
steps also require positive `vector_bytes<=64` that is a multiple of the
element width and divides the step length; DMA steps omit that field.
`wait="after-each-segment"` and `completion` naming the last id make ordering
and completion explicit. Each DMA waits before the next step, and the copy
task completes only after the entire sequence, including any RVV tail.

A shared DMA body requires the two row addresses to have the same residue
modulo 64. Different or unknown residues select all-RVV for that row, with
natural element alignment and valid ranges still proven. The planner
chooses every step, and the verifier checks exact gap-free, nonoverlapping
coverage in both endpoints. The emitter does not discover a route or expand
the external ABI. RVV accesses only valid elements through correct VL/mask
lowering; whole-register accesses cannot replace these element operations.
Internal padding that participates in computation retains its original
initialization. This extension does not add transpose/packing semantics.
The current NR target does not permit the required vle16/vse16 forms, so
16-bit copies needing an RVV segment are rejected explicitly; legal
all-DMA 16-bit copies remain supported. This is an implementation limit.

The 64-byte rule is the DMA platform contract in `SpmDma.h`, not a universal
AME/RVV memory rule. The matrix specification in
`plan/riscv-matrix-spec_32bit-开芯院.pdf`, section 4.3 (printed pages 17–19),
defines ordinary tile accesses and byte row strides separately from whole
matrix accesses. `reference/src/cases/nh_e6_ame_gemm/src/main.S:106–125`
uses M2/N3/K4, A/B stride 4 B and C stride 12 B, with 256-byte-aligned bases.
`reference/src/cases/nh_e6_dma_spm/src/main.c:355–395` demonstrates ordinary
RVV SPM access and a small AME tile inside 64-byte DMA buffers; lines
402–407 initialize complete SPM lines. These references support small
tiles/strides and SPM reachability, but do not prove arbitrary unaligned
bases or constitute board validation of the new boundary routes. Conservative
AME/pointwise planner restrictions remain implementation limits.

Descriptor offsets are in elements; transfer tile/segment offsets and lengths,
byte pitches and window offsets are in bytes. Layout strides remain in elements. A subview adds offsets using source strides,
a reinterpret cast replaces the descriptor offset, and a byte view changes
the aligned pointer separately. These terms must not be applied twice. An
unranked argument supplies no allocation extent: validity of the frontend's
effective view remains a frontend/caller promise.

Argument alignment and disjointness are separate facts. `contract.abi` applies
only to functions with no symbol uses in the module; helpers cannot inherit
entry promises when callers can pass aliases or unaligned views. Emission
records `bosc_ame.dma.abi="args_aligned64"` for alignment alone, or
`"args_aligned64_nonalias"` when disjointness is also promised. Alignment alone
permits an ordered DDR/SPM transfer but does not prove concurrent DDR accesses
disjoint; the DMA lifetime checker retains its alias checks.

The implemented emission subset is deliberately bounded:

- AME accepts static signed i8 × i8 → i32 matmul with one M tile, full N/K
  subtiles and compatible physical layouts. Each matmul loads and stores C;
  accumulator registers do not remain live across outer K iterations or DMA
  runtime calls.
- Other supported buffer-form matmuls use an explicit compatibility recipe:
  `plan.compute = {engine = "scalar", kind = "matmul", order = "mnk",
  shape = array<i64: m, n, k>, fp_policy = "preserve", operands = [...]}`.
  Emission creates serial M/N/K loops and clones the original scalar casts,
  multiply and add payload without reassociation. It keeps a separate floating
  addition outside the matmul separate. DDR elements use scalar MemRef access;
  SPM elements use `vector<1>` load/extract and broadcast/store operations,
  preserving the target's memory-access rules. Operand bindings and static
  bounds are verified before expansion. This bounded recipe uses the shared
  planner and does not add target operations to the neutral input contract.
- Pointwise generic uses equal-shaped contiguous rank-one buffers, identity
  parallel indexing, a pure supported scalar expression and a direct yield.
  Mixed integer widths and f32 values use one common lane count. An exact scalar
  constant capture has an explicit
  `plan.vector.scalar_operands` entry and becomes a vector broadcast inside the
  planned loop, before its arithmetic operation. Other scalar captures require
  scalar-load/SSA/fill broadcasts;
  unknown captures remain invalid.
- Ordered f32 addition reduction uses fixed vector chunks and a private host
  rank-zero accumulator, with no fastmath. Its rounding may differ from a
  frontend's other reduction tree; LayerNorm retains its established
  `rtol=atol=2e-5` acceptance, rather than claiming bitwise equivalence.
- Exact f32 max-number reduction uses the same ordered chunk traversal and
  explicit initializer. One NaN selects the numeric argument, two NaNs produce
  NaN, and signed-zero ties may select either sign; the target and simulator
  choose +0 for mixed signs. Padding is actual initialized data, never an
  implicit ignored lane: Softmax uses -Inf for both invalid input lanes and
  the max initializer. Exp(-Inf)=0 then contributes zero to the ordered sum.
- Host rank-zero allocations must be static, private and outside repeated
  scopes. Emission creates stack storage and removes its heap deallocation.
- Physical overlap between different logical roots is checked independently
  of logical noalias. Without a general liveness reuse proof, only mutually
  exclusive arms may reuse overlapping windows. Capacity uses actual window
  high-water marks. Every SPM root and static view alias must remain local and
  use a supported synchronous memory operation in `func.func`, `scf.for` or
  `scf.if`. Allocation inside a concurrent region or capture by its users is
  rejected, including when the logical graph remains incomplete. Returning or
  yielding the buffer, extracting its address, dynamic aliases and unmodeled completion
  prevent automatic or fixed SPM placement.
- A function with private or fixed SPM windows cannot contain a function call.
  V1 has no interprocedural workspace contract, so even a call without memref
  arguments could clobber a live window through its own SPM placement. Planned
  input and emission enforce the same restriction. Such a function also cannot
  have module-local symbol uses: a DDR-only caller could invoke it concurrently,
  and indirect references do not prove serial ownership. This restriction also
  applies to serial calls until V1 has that proof. Calls between functions using
  only DDR remain legal.

### Explicit mathematical accuracy

Approximation requires an explicit per-function value contract. For example,
the two ViT nonlinear examples export these independent authorizations:

```mlir
// GELU's function:
contract.math = {erf = {domain = "all-f32", special_values = "ieee",
  max_abs_error = 2.0e-6 : f64, max_rel_error = 0.0 : f64}}
// Softmax's function:
contract.math = {exp = {domain = "all-f32", special_values = "ieee",
  max_abs_error = 1.1210387714598537e-44 : f64,
  max_rel_error = 2.0e-6 : f64}}
```

The finite-result allowance is `absolute + relative * |mathematical result|`.
All f32 inputs, including NaN, infinities, signed zero and subnormals, belong
to the domain. IEEE value behavior uses round-to-nearest, ties-to-even and
gradual underflow; NaN payloads and exception flags are outside this value
contract. V26 rejects malformed, misplaced, negative and nonfinite budgets.
Absence of an authorization retains the existing mathematical operation and
the backend diagnoses an unsupported recipe. No target recipe or SFU type
appears in this contract, and other frontends can supply the same attributes.

The BOSCAME planner records a recipe per math operation in `plan.vector.math`,
including its payload index, mathematical function, recipe version, evaluation
order and guaranteed error. The verifier matches each record against the
actual operation and the neutral authorization, including no-fastmath and f32
requirements; missing, duplicated or forged records are rejected before
emission. Unrelated arithmetic retains its original order.

For the restricted FPGA profile, these expressions use `engine="scalar"`,
`lanes=1`, `vector_bytes=4` and `fp_policy="explicit-approximation"`.
Emission loads/extracts one vector lane, evaluates the selected polynomial
with scalar f32 arithmetic, then broadcasts/stores one lane. It makes no host
math call. This uses verified target instructions without expanding the RVV
whitelist or creating vector-register lifetimes across DMA calls. Ordinary
pointwise expressions and sum/max reductions retain their vector recipes.

Erf uses generated piecewise polynomials; exp uses ln(2) range reduction and
two power-of-two scaling factors, retaining subnormal exponentials. The GELU
source stays in erf form, and Softmax retains max subtraction, exp, ordered
sum and division. The [numerical evidence](validation/nonlinear-math/README.md)
documents coefficient generation, analytic error bounds, independent sampling
and why the SFU exp2 route is not yet selected. The existing `2e-5` output
tolerances remain unchanged; this is an explicit numerical change, not a
claim of bitwise agreement or whole-model accuracy.

Unsupported shapes, payloads, addresses or capacity combinations diagnose the
actual failed constraint. Emission does not select a new route after validation
fails. The existing strict distinction between DDR, SPM and registers remains
in the dialect verifiers and functional simulator.

## Verification evidence

Native route tests run the three neutral contract fixtures through the actual
planner and emitter. Their checked workspace and surviving task counts are:

| Contract fixture | SPM windows | Workspace bytes | Surviving tasks |
| --- | ---: | ---: | ---: |
| 196 × 768 × 768 i8 matmul | 2 | 20480 | 10 |
| 128 × 1024 × 128 i8 matmul | 0 | 0 | 1 |
| 197 × 768 LayerNorm | 6 | 24576 | 21 |

The 196-row contract uses SPM A/C and direct B. The 128-row contract folds five
source tasks into one direct matmul. LayerNorm also retains two host rank-zero
buffers. Tests cover mixed routes, safe capacity fallback, failed alternatives,
source mappings and malformed plans. Strict functional simulation covers
nonzero signed accumulation, i32 wrap, tail-before-full execution, polluted SPM,
DDR guards, unchanged A/B and LayerNorm tolerance.

The regenerated production examples also complete `make`, `make check` and
`make simcheck`. Their actual routes and final linked-image stack bounds are:

| Production example | Routes | Physical SPM windows / bytes | E6 stack bound |
| --- | --- | ---: | ---: |
| 196 × 768 × 768 i8 matmul | SPM A/C, direct DDR B | 2 / 20480 | 1216 B |
| 128 × 1024 × 128 i8 matmul | Direct DDR A/B/C | 0 / 0 | 656 B |
| 197 × 768 LayerNorm | SPM/DMA vectors, host scalars | 5 / 20480 | 1168 B |

The production LayerNorm's bufferized graph reuses its mean/invstd broadcast
root and therefore needs one fewer window than the handwritten fixture.
Its 21 tasks replace the integrated baseline's 19 because scalar loads and
broadcast materialization are explicit; the existing tolerance is unchanged,
and all recorded output bytes match that baseline. The integer examples
compare exactly, including the complete 196-row signed/wrap stress suite.

The [validation record](BOSCAMEKernelContractValidation.md) gives traffic,
hashes, baseline reconstruction limits and the reproducible linked-ELF stack
audit, including DMA failure hooks and the trap path. Every stack bound is
below the unchanged 65536-byte limit. These example checks are separate from
repository-wide `check-buddy` and formatting. Historical board results linked
from the earlier experimental documentation apply to their recorded compiler
outputs; simulator elapsed time is not hardware performance evidence.

The [zero-accumulator/W8A8 validation](validation/zero-acc-w8a8/README.md)
records first-K initialization, emitted mixed-width pointwise expressions,
scalar scale boundaries and the default ViT dense rewrite. Kernel correctness,
graph ABI evidence and complete-model accuracy are reported independently.
