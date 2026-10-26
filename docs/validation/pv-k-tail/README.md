# PV reduction-tail validation

This checkpoint builds the unchanged real Triton source
`attention_pv_12x197x64x197/kernel.py` through the neutral contract frontend,
contract verification, planning, plan verification, emission, LLVM translation,
restricted NR assembly and RISC-V object generation. It supplements the
historical [37/40 W8A8 checkpoint](../zero-acc-w8a8/); the other 39 entries were
not rebuilt for this single-case checkpoint.

The [full-grid report](simcheck-report.json) is **PASS** for all three rounds:
1872 executed programs and 453888 f32 output comparisons, bit for bit, with
maximum absolute error 0. Current collection independently revalidates all
48 shard identities, actual program sequences and output hashes.

The frontend retains two M branches. Each executes three complete K=64 loop
iterations and a fixed block-3 tail with five valid K elements. Arithmetic
still uses A=16x64 and B=64x16. Every block computes a zero-initialized product,
then separately adds that product to the running result. Positive-zero tail
padding, all 64 reduction steps, multiply/add rounding and block-add order
remain intact. There is no floating AME MMA or newly contracted FMA.

| Access | Selected route and actual valid geometry |
| --- | --- |
| Full-M/full-K A | Raw ACC32 matrix copy: 16 rows, 256 B per row, DDR pitch 788 B, SPM pitch 256 B |
| Full-M/tail-K A | Same route: 16 rows, 20 B per row, unchanged pitches |
| Tail-M/full-K A | Same route: 5 rows, 256 B per row, unchanged pitches |
| Tail-M/tail-K A | Same route: 5 rows, 20 B per row, unchanged pitches |
| B | Existing legal DMA: 197 packets/program, 64 B per packet, DDR pitch 256 B; tail transfers five rows |
| C | DMA writes only: 16 valid rows or five tail rows, 64 B per row; no old external C read |

ACC32 moves raw f32 bits between DDR and SPM. The 788 B A pitch prevents the
existing DMA route; planner reason is
`dma-illegal-exact-acc32-rectangle`. B/C retain
`bounded-external-boundary` DMA. Planned bank high-water marks are 10240 B and
9216 B. The regenerated graph is complete with five scopes and 29 tasks.

[Build report](build-report.json), [static route audit](route-summary.json)
and [artifact/source hashes](artifact-hashes.json) describe the actual build.
[capture.py](capture.py) reads those artifacts and asserts the recorded
prefix/tail arithmetic, matrix geometry, graph and absence of FMA. Its byte
totals are static expectations independently checked by strict simulation.
The [production object audit](production-object-audit.json) passes with 3177
instructions, including 20 AME transfer instructions and 78 RVV instructions.

The [first-program probe](first-program-probe.json) runs one full-M and one
tail-M program and checks outputs bit for bit. The
[parallel-worker probe](shard-probe.json) runs two selected full-M/tail-M
subsets through all three rounds. These probes are explicitly partial and
do not establish full-grid execution.

The full-grid harness partitions only validation work into 48 independent
`(head, N tile)` shards. Every shard interprets all 13 actual M programs with
the unchanged strict simulator and retains one Machine/SPM across all three
rounds. SPM is poisoned only on its first construction. Each launch audits
exact logical DDR rows, transfer geometry, readonly A/B, guards and writes
outside the selected output tiles. The parent reconstructs outputs solely
from the returned executed slices, rejects duplicate/missing programs, and
requires complete 624-program/151296-output coverage and uint32 equality to
the independent oracle in each round.

The normal round visits M=0..12 within each shard. Both stress tail-first and
stress reverse visit M=12..0 because head and N tile are fixed. The third
round repeats the same stress and reverse-M sequence on retained dirty SPM.
No global order across shards, or single-Machine global reverse order, is
claimed. Stress inputs distinguish block-local accumulation from reassociation
across K blocks. Old C contains NaNs and must not enter the computation.

Each round executes 624 programs and compares 151296 outputs. Actual DDR
traffic is 7451328 B from A, 7867392 B from B and 605184 B to C, totaling
15923904 B. There are 2496 A matrix copies: 1728 full-M/full-K, 576
full-M/tail-K, 144 tail-M/full-K and 48 tail-M/tail-K. B performs 122928 DMA
packets; including C, the round performs 132384 DMA transfers and 170208
external DDR accesses. Full-M/tail-M program totals are 26240/16868 B.
The normal output SHA-256 is
`5ff2e2fba234e34b9c290feaf6e6bf7ab00df82bb9e7b68d2bef511d72548cb4`;
both stress rounds produce
`5692f1d27ac16a9b7b70ae63e18c675b4206b1b3dc117a6b65e85b581e44f8b9`.

The PV oracle was corrected to match the original floating frontend's
zero-product plus separate block-add expansion. The previous whole-K sequential
reference incorrectly reassociated multiple K blocks; PV had not passed that
reference. This changes the test reference, not compiler floating semantics.
QK's single-block reference and historical evidence remain unchanged.

[Simulator source provenance](simulator-source-provenance.json) records the
source used when the 48-worker run began, the current source, and unchanged
runtime-function hashes. Subsequent edits strengthen only collection checks
and negative tests. Current collection requires each shard's fixed identity
and exact recorded program sequence as well as independent output hashes.

Final `check-buddy` passes: examples contain 239 PASS and two unsupported
tests; regressions contain 622 PASS, two unsupported tests and one expected
failure, with no unexpected failures. The final focused run passes all four
K-tail, PV-oracle/coverage, reaching-zero-loop-copy and immutable-zero-copy
tests. New oracle/coverage checks distinguish reassociation and FMA, retain
full zero-padded BK64 arithmetic, reject logical-row violations and writes
outside authorized tiles, and reject missing/duplicate/incorrectly ordered
shard evidence. Changed-file formatting and whitespace checks pass.

Reproduce from the repository root, using a Python environment with Triton and
NumPy:

```sh
PYTHONPATH=build/python_packages python -B examples/FPGA-BOSCAME/vit-base-patch16-224/tools/build.py --case attention_pv_12x197x64x197 --stage asm
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B examples/FPGA-BOSCAME/vit-base-patch16-224/tools/simcheck.py --case attention_pv_12x197x64x197 --pv-workers 48
python -B examples/FPGA-BOSCAME/vit-base-patch16-224/tools/simcheck.py --case attention_pv_12x197x64x197 --collect
python -B docs/validation/pv-k-tail/capture.py
python -B tests/Conversion/TritonToLinalgExperimental/vit-pv-oracle.py
python examples/FPGA-BOSCAME/tools/check_nr_elf.py examples/FPGA-BOSCAME/vit-base-patch16-224/build/kernels/attention_pv_12x197x64x197/kernel.o --objdump llvm/build/bin/llvm-objdump
```

These are emitted-IR simulations and assembly/object audits. They do not
execute the assembly, run an FPGA, measure hardware performance or establish
complete-model accuracy. Probe durations measure interpreter cost only.
