// RUN: buddy-opt %s --verify-diagnostics --split-input-file

// The SPM storage space can only name one of the two SPM banks.  The attribute
// is verified while the type is parsed, so an out-of-range bank is rejected
// before any pass runs.
// expected-error @+1 {{SPM bank must be in [0, 1], got 2}}
func.func @bad_bank() -> memref<64xi8, #bosc_ame.spm<2>> {
  %u = "test.undefined"() : () -> memref<64xi8, #bosc_ame.spm<2>>
  return %u : memref<64xi8, #bosc_ame.spm<2>>
}

// -----

// The same holds for the DMA channel carried by a token type.
// expected-error @+1 {{DMA channel must be in [0, 1], got 3}}
func.func @bad_token(%t: !bosc_ame.dma_token<3>) {
  return
}
