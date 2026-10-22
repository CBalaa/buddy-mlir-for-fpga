// One row of the 197x768 LayerNorm launch, after explicit bufferization.
// The caller slices X/Out to the chosen row using aligned pointers and zero
// descriptor offsets. Weight and Bias are shared across rows. All four DDR
// arguments expose exactly 768 f32 elements (3072 bytes), never the padding.
// Distinct arguments are nonalias and their aligned pointers are 64B aligned.
//
// Preserve the original TTIR computation: X has 1024 logical lanes, with
// masked-load other=0. Weight other=1 and Bias other=0 are explicit fills.
// Both sums cover all 1024 lanes and divide by 768. The centered select zeros
// padded X lanes before the variance reduction. Only 768 lanes store.
// Ordinary aligned scratch buffers make padding and extra staging explicit.

#id = affine_map<(d0) -> (d0)>
#scalar = affine_map<(d0) -> ()>

module attributes {bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  func.func @layernorm_program(%x: memref<768xf32>,
                               %weight: memref<768xf32>,
                               %bias: memref<768xf32>,
                               %out: memref<768xf32>) {
    %zero = arith.constant 0.0 : f32
    %one = arith.constant 1.0 : f32
    %width = arith.constant 768.0 : f32
    %valid_width = arith.constant 768 : index
    %epsilon = arith.constant 1.0e-12 : f32
    %xpad = memref.alloca() alignment = 64 : memref<1024xf32>
    %wpad = memref.alloca() alignment = 64 : memref<1024xf32>
    %bpad = memref.alloca() alignment = 64 : memref<1024xf32>
    %centered = memref.alloca() alignment = 64 : memref<1024xf32>
    %squares = memref.alloca() alignment = 64 : memref<1024xf32>
    %result = memref.alloca() alignment = 64 : memref<1024xf32>
    %sum = memref.alloca() alignment = 64 : memref<f32>
    %xvalid = memref.subview %xpad[0] [768] [1] : memref<1024xf32> to memref<768xf32, strided<[1]>>
    %wvalid = memref.subview %wpad[0] [768] [1] : memref<1024xf32> to memref<768xf32, strided<[1]>>
    %bvalid = memref.subview %bpad[0] [768] [1] : memref<1024xf32> to memref<768xf32, strided<[1]>>
    %rvalid = memref.subview %result[0] [768] [1] : memref<1024xf32> to memref<768xf32, strided<[1]>>
    linalg.fill ins(%zero : f32) outs(%xpad : memref<1024xf32>)
    linalg.fill ins(%one : f32) outs(%wpad : memref<1024xf32>)
    linalg.fill ins(%zero : f32) outs(%bpad : memref<1024xf32>)
    linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
      ins(%x : memref<768xf32>) outs(%xvalid : memref<768xf32, strided<[1]>>) {
      ^bb0(%value: f32, %old: f32):
        linalg.yield %value : f32
    }
    linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
      ins(%weight : memref<768xf32>) outs(%wvalid : memref<768xf32, strided<[1]>>) {
      ^bb0(%value: f32, %old: f32):
        linalg.yield %value : f32
    }
    linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
      ins(%bias : memref<768xf32>) outs(%bvalid : memref<768xf32, strided<[1]>>) {
      ^bb0(%value: f32, %old: f32):
        linalg.yield %value : f32
    }
    linalg.fill ins(%zero : f32) outs(%sum : memref<f32>)
    linalg.reduce ins(%xpad : memref<1024xf32>) outs(%sum : memref<f32>) dimensions = [0]
      (%value: f32, %acc: f32) {
        %next = arith.addf %value, %acc : f32
        linalg.yield %next : f32
    }
    %total = memref.load %sum[] : memref<f32>
    %mean = arith.divf %total, %width : f32
    linalg.generic {indexing_maps = [#id, #scalar, #id], iterator_types = ["parallel"]}
      ins(%xpad, %mean : memref<1024xf32>, f32) outs(%centered : memref<1024xf32>) {
      ^bb0(%value: f32, %average: f32, %old: f32):
        %difference = arith.subf %value, %average : f32
        %lane = linalg.index 0 : index
        %valid = arith.cmpi slt, %lane, %valid_width : index
        %masked = arith.select %valid, %difference, %zero : f32
        linalg.yield %masked : f32
    }
    linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
      ins(%centered : memref<1024xf32>) outs(%squares : memref<1024xf32>) {
      ^bb0(%value: f32, %old: f32):
        %square = arith.mulf %value, %value : f32
        linalg.yield %square : f32
    }
    linalg.fill ins(%zero : f32) outs(%sum : memref<f32>)
    linalg.reduce ins(%squares : memref<1024xf32>) outs(%sum : memref<f32>) dimensions = [0]
      (%value: f32, %acc: f32) {
        %next = arith.addf %value, %acc : f32
        linalg.yield %next : f32
    }
    %square_total = memref.load %sum[] : memref<f32>
    %variance = arith.divf %square_total, %width : f32
    %shifted = arith.addf %variance, %epsilon : f32
    %inverse = math.rsqrt %shifted : f32
    linalg.generic {indexing_maps = [#id, #scalar, #id, #id, #id], iterator_types = ["parallel"]}
      ins(%centered, %inverse, %wpad, %bpad : memref<1024xf32>, f32, memref<1024xf32>, memref<1024xf32>) outs(%result : memref<1024xf32>) {
      ^bb0(%value: f32, %scale: f32, %w: f32, %b: f32, %old: f32):
        %normalized = arith.mulf %value, %scale : f32
        %weighted = arith.mulf %normalized, %w : f32
        %affine = arith.addf %weighted, %b : f32
        linalg.yield %affine : f32
    }
    linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
      ins(%rvalid : memref<768xf32, strided<[1]>>) outs(%out : memref<768xf32>) {
      ^bb0(%value: f32, %old: f32):
        linalg.yield %value : f32
    }
    return
  }
}
