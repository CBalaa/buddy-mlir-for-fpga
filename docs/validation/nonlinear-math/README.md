# Explicit f32 erf / exp implementations

This evidence directory concerns the two explicitly authorized ViT GELU and
Softmax kernels. The neutral `contract.math` records mathematical function,
`all-f32` domain, `ieee` value semantics and the mixed error allowance
`absolute + relative * |reference|`. It does not name a target instruction or
recipe. The BOSCAME planner selects a concrete recipe, the verifier recomputes
its authorization and the emitter expands it. An absent or tighter contract
continues to reject these approximations. Other `math.erf` / `math.exp`
operations are not rewritten by a global math pass.

The retained acceptance thresholds are `atol=2e-5, rtol=2e-5`. This directory's
mathematical certificates, coefficient generation and sampled evidence are
independent of the full-grid compiler/simulator validation recorded by the ViT
validation report. Nothing here establishes whole-model accuracy or board
performance.

## Reproduction

From the repository root, with Python containing NumPy, SciPy and mpmath:

```sh
python docs/validation/nonlinear-math/generate.py \
  --write-header --quick \
  --report docs/validation/nonlinear-math/error-certificate.json
python docs/validation/nonlinear-math/generate.py \
  --report docs/validation/nonlinear-math/numerical-evidence.json
python tests/Conversion/LowerLinalgToBOSCAME/math-approx-numerics.py
```

Without `--write-header`, the generator checks that every generated coefficient
and the generated header match exactly. `--quick` needs only NumPy; it still
recomputes every rational enclosure and forward error certificate. The default
dense run uses 1,000,001 initial points per function, adds boundary neighbors,
and separately checks 90-digit mpmath references. The compiler regression
`PlanMathApprox.py` checks the actual planned/emitted route and target assembly;
the numerical regression also covers 20,000 random binary32 encodings. The
NumPy evaluator is evidence for the exact expression selected by the emitter;
it is not a runtime implementation or simulator fallback.

The coefficient header is generated using exact rational intervals. Machin's
formula `pi = 16 atan(1/5) - 4 atan(1/239)` and alternating series bound pi; an
integer square root bounds `2/sqrt(pi)`. Positive exp Taylor sums plus a
geometric remainder bound exp; their reciprocals bound negative exp. An
alternating Maclaurin sum bounds erf at each dyadic center. No floating-point
reference library supplies the proof's coefficients or analytic bounds. Decimal
upper bounds in the report are rounded upward from the rational result.

## erf recipe

`erf.f32.piecewise-taylor.v1` uses odd symmetry. For `|x|<0.5`, it evaluates
`x * sum[n=0..7] (2/sqrt(pi))*(-1)^n*x^(2n)/(n!*(2n+1))`. This avoids
cancellation near zero and retains subnormal results. On `[0.5,4]`, seven
half-unit intervals use degree-11 Taylor polynomials with centers
`0.75, 1.25, ..., 3.75`. Coefficients and centers are selected per lane before
one Horner evaluation. Center subtraction is exact by Sterbenz's lemma.
For `|x|>=4`, the result is the signed unit value; the omitted tail is bounded
by `erfc(4) <= exp(-16)/(4*sqrt(pi)) < 1.588e-8`.

For each interval, the Taylor remainder is bounded by
`2/sqrt(pi) * exp(-a*a) * abs_coeff_bound(H_11,b) * (1/4)^12 / 12!`, where
`[a,b]` is the interval and `H_11` is the physicists' Hermite polynomial.
The coefficient absolute sum is intentionally conservative. The odd branch
uses its alternating-series remainder and separately accounts for rounding
`x*x`. The resulting full finite-domain absolute bound, including coefficient
quantization and independent binary32 multiplies/adds, is **1.138e-7**. The
public mathematical allowance used for this recipe is **2e-6 absolute**.

`erf(+0)=+0`, `erf(-0)=-0`, `erf(+Inf)=1`, `erf(-Inf)=-1`, and NaN produces
NaN. All finite binary32 values are accepted. This retains the original
`erf(x/sqrt(2))` GELU expression and its multiplication order; there is no tanh
substitution. In particular the source expression produces NaN for GELU of
negative infinity, and this behavior is retained.

## exp recipe

`exp.f32.range-reduced-taylor.v1` first bounds eagerly evaluated inputs and
replaces NaNs before any float-to-integer conversion. It computes
`k=trunc(fl(fl(x*log2(e))+copysign(0.5,x)))` and then
`r=fl(fl(x-k*ln2_hi)-fl(k*ln2_lo))`, with `ln2_hi=355/512` and
`ln2_lo=f32(ln(2)-ln2_hi)`. The high product is exact over the proved integer
range. The generator bounds all rounding in choosing k and proves the true
reduced argument is within **0.346588**, and the evaluated argument within
**0.347**. A degree-7 Taylor polynomial evaluates exp(r). This Taylor series is
used only on the reduced interval.

Scaling constructs two normal powers of two by integer exponent bits:
`k1=clamp(k,-126,127)`, `k2=k-k1`. The result is
`fl(fl(P(r)*2^k2)*2^k1)`. `k2` is in `[-24,1]`; the first product remains
normal. The final product is the only rounding to subnormal, so this route does
not flush small exponentials to zero. The normal-result relative error bound
is **2.583e-7**; the subnormal absolute bound is **3.737e-45**. The declared
recipe allowance is **2e-6 relative plus 8 minimum subnormals absolute**
(`2^-146 = 1.1210387714598537e-44`). No small term is dropped before the sum.

The first overflowing binary32 input is `88.72283935546875`; the previous
input `88.72283172607422` remains finite even after accounting for the full
approximation error. The last binary32 input rounding to zero is
`-103.97208404541016`; its successor `-103.97207641601562` produces the minimum
positive subnormal. The generator proves these comparisons using rational
exp enclosures against the overflow midpoint `2^128-2^103` and the underflow
midpoint `2^-150`. Values beyond these thresholds, including infinities, are
selected explicitly. `exp(+0)=exp(-0)=1`, `exp(-Inf)=0`, `exp(+Inf)=+Inf`, and
NaN produces NaN. Every eagerly evaluated `fptosi` operand is finite with
magnitude below 151; none depends on a later select to eliminate poison.

## Rounding and composed outputs

All generated operations use binary32 round-to-nearest, ties-to-even, gradual
underflow and separate multiply/add. The helper sets no fast-math or contract
flags. There is no FMA, host math call or math operation remaining in a recipe.
NaN payload/signaling behavior and exception flags are outside the value
contract; the special output classifications above are part of it.

The proof uses `u=2^-24` and `eta=2^-150`, so an elementary rounded operation
has absolute error at most `u*|exact result|+eta`. Horner propagation includes
all coefficient quantization, intermediate multiplication/addition errors,
range-reduction errors, final scaling, and gradual underflow. The reported
**analytic bounds** are derived from rational enclosures. The roughly one
million samples per function and mpmath boundary checks are separate
**numerical observations**, not proofs of an exhaustive bound or correct
rounding.

For composition, the reference is the source primitive order with erf/exp
computed mathematically and rounded to f32. It is not an assertion that NumPy's
`exp(dtype=float32)` is always correctly rounded. The unchanged existing
NumPy/Python oracle and a separate high precision mathematical reference are
checked by the full-grid simcheck.

For finite GELU inputs, `|erf argument|<4` implies `|half_x|<3`. Propagating
the erf error, the reference call's f32 rounding, and both final arithmetic
steps gives **1.862e-6 absolute**. Outside this interval, the true erf itself
rounds to exactly the signed unit used by this recipe, so the retained source
expression agrees. This also covers the cancellation in the GELU negative
tail; it does not replace the source expression with an erfc formulation.

For finite Softmax rows, a selected maximum guarantees at least one exp(0)=1;
exp values lie in `[0,1]`, and the exact sum lies in `[1,256]`. The sum preserves
source order and includes padded exp(-Inf)=0 values. The positive-sum rounding
bound is `gamma_255=255*u/(1-255*u)`. Including the exp approximation, reference
exp rounding, both ordered sums and both normalization roundings gives a
relative comparison bound **3.116e-5** for normal probabilities. Since the
output is at most one, `3.116e-5*p <= 2e-5+2e-5*p` for every `p` in `[0,1]`.
For ideal probabilities below `2^-100`, numerator and sum bounds instead give
absolute error below `2^-90`, including subnormal exp and division rounding.
This separates tiny results from the normal relative-error argument. The
allowance is sufficient for the existing threshold without changing it.

Special rows require value classification, rather than the finite-row bound:
finite rows containing negative infinity give zero at those positions; a row
with NaN, positive infinity, or only negative infinities produces NaNs under
the retained subtraction/exp/sum/division algorithm. Max itself is exact and
its mask/NaN/signed-zero semantics are verified separately.

## SFU exp2 assessment

`reference/SFU 拓展指令.md` describes `vsfu.exp2.v` encoding and a fractional
LUT plus quadratic evaluator. Its maximum error of 2 ULP is explicitly a design
target, not measured evidence. The document says subnormal inputs are treated
as zero, and its estimate does not follow IEEE rounding-mode/exception-flag
behavior. The repository's current math route has no verified chain joining
that encoding, an exp2 lowering and matching LUT simulator semantics. Therefore
this change uses the polynomial recipe. A future `exp2(x*log2(e))` route must
also budget the product/range error and provide implementation-specific
coefficients and simulator evidence; the SFU document alone is insufficient.
No other SFU instructions are introduced.
