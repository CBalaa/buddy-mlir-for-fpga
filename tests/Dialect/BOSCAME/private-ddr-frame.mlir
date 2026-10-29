// RUN: buddy-opt %s --verify-diagnostics --split-input-file

module attributes {bosc_ame.target = "nr-fpga"} {
  func.func @private_accumulator() {
    %frame = bosc_ame.ddr.frame : memref<512xi8>
    %shift = arith.constant 64 : index
    %view = memref.view %frame[%shift][] : memref<512xi8> to memref<2x16xi32>
    %acc = bosc_ame.tile.load_acc %view {bosc_ame.direct_ddr} : memref<2x16xi32> -> vector<2x16xi32>
    bosc_ame.tile.store_acc %acc, %view {bosc_ame.direct_ddr} : vector<2x16xi32>, memref<2x16xi32>
    bosc_ame.ddr.release %frame : memref<512xi8>
    return
  }
}

// -----
module attributes {bosc_ame.target = "nr-fpga"} {
  func.func @ordinary_stack_is_not_frame() {
    %stack = memref.alloca() alignment = 64 : memref<2x16xi32>
    // expected-error @+1 {{direct DDR tile endpoint must trace to an external function entry argument or a managed private DDR frame}}
    %acc = bosc_ame.tile.load_acc %stack {bosc_ame.direct_ddr} : memref<2x16xi32> -> vector<2x16xi32>
    return
  }
}

// -----
module attributes {bosc_ame.target = "nr-fpga"} {
  func.func @out_of_frame() {
    // expected-error @+1 {{private DDR endpoint has an unproved view bound or exceeds frame capacity}}
    %frame = bosc_ame.ddr.frame : memref<128xi8>
    %shift = arith.constant 64 : index
    %view = memref.view %frame[%shift][] : memref<128xi8> to memref<2x16xi32>
    %acc = bosc_ame.tile.load_acc %view {bosc_ame.direct_ddr} : memref<2x16xi32> -> vector<2x16xi32>
    bosc_ame.ddr.release %frame : memref<128xi8>
    return
  }
}

// -----
module {
  func.func @cpu_only_out_of_frame() {
    // expected-error @+1 {{private DDR endpoint has an unproved view bound or exceeds frame capacity}}
    %frame = bosc_ame.ddr.frame : memref<128xi8>
    %shift = arith.constant 128 : index
    %zero = arith.constant 0 : index
    %view = memref.view %frame[%shift][] : memref<128xi8> to memref<1xi32>
    %value = memref.load %view[%zero] : memref<1xi32>
    bosc_ame.ddr.release %frame : memref<128xi8>
    return
  }
}

// -----
module {
  func.func @dynamic_frame_offset(%shift: index) {
    // expected-error @+1 {{private DDR endpoint has an unproved view bound or exceeds frame capacity}}
    %frame = bosc_ame.ddr.frame : memref<128xi8>
    %zero = arith.constant 0 : index
    %view = memref.view %frame[%shift][] : memref<128xi8> to memref<1xi32>
    %value = memref.load %view[%zero] : memref<1xi32>
    bosc_ame.ddr.release %frame : memref<128xi8>
    return
  }
}

// -----
module {
  func.func @parallel_frame_access() {
    // expected-error @+1 {{private DDR frame may not escape its serial invocation}}
    %frame = bosc_ame.ddr.frame : memref<128xi8>
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    scf.parallel (%i) = (%zero) to (%one) step (%one) {
      %value = memref.load %frame[%i] : memref<128xi8>
      scf.reduce
    }
    bosc_ame.ddr.release %frame : memref<128xi8>
    return
  }
}

// -----
module {
  func.func @generic_frame_free() {
    // expected-error @+1 {{managed private DDR storage requires ddr.release, not memref.dealloc}}
    %frame = bosc_ame.ddr.frame : memref<128xi8>
    memref.dealloc %frame : memref<128xi8>
    bosc_ame.ddr.release %frame : memref<128xi8>
    return
  }
}

// -----
module {
  func.func @frame_escape() -> memref<128xi8> {
    // expected-error @+1 {{requires exactly one release immediately before the function return}}
    %frame = bosc_ame.ddr.frame : memref<128xi8>
    return %frame : memref<128xi8>
  }
}

// -----
module {
  func.func private @unknown(memref<128xi8>)
  func.func @unknown_user() {
    // expected-error @+1 {{private DDR pointer escapes or has an unsupported memory user: func.call}}
    %frame = bosc_ame.ddr.frame : memref<128xi8>
    func.call @unknown(%frame) : (memref<128xi8>) -> ()
    bosc_ame.ddr.release %frame : memref<128xi8>
    return
  }
}

// -----
module {
  func.func @too_large() {
    // expected-error @+1 {{requires a positive aligned static i8 frame of at most 256 KiB}}
    %frame = bosc_ame.ddr.frame : memref<262144xi8>
    bosc_ame.ddr.release %frame : memref<262144xi8>
    return
  }
}

// -----
module attributes {bosc_ame.target = "nr-fpga"} {
  func.func @single_row(%arg0: memref<1x40xi32, strided<[1000, 1]>>) {
    %frame = bosc_ame.ddr.frame : memref<256xi8>
    %zero = arith.constant 0 : index
    %dense = memref.view %frame[%zero][] : memref<256xi8> to memref<40xi32>
    %view = memref.reinterpret_cast %dense to offset: [0], sizes: [1, 40], strides: [1000, 1] : memref<40xi32> to memref<1x40xi32, strided<[1000, 1]>>
    %acc = bosc_ame.tile.load_acc %view {bosc_ame.direct_ddr} : memref<1x40xi32, strided<[1000, 1]>> -> vector<1x40xi32>
    bosc_ame.tile.store_acc %acc, %view {bosc_ame.direct_ddr} : vector<1x40xi32>, memref<1x40xi32, strided<[1000, 1]>>
    bosc_ame.ddr.release %frame : memref<256xi8>
    return
  }
}
