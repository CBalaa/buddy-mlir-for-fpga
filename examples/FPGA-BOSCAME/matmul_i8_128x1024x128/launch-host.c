#include "support.h"
#define M 128
#define N 1024
#define K 128
extern void _mlir_ciface_kernel_matmul_i8_128x1024x128(MemRef2 *, MemRef2 *,
                                                       MemRef2 *);
int launch(void) {
  int8_t *a = workspace(0), *b = workspace((size_t)M * K + 64);
  int32_t *c = workspace((size_t)M * K + (size_t)N * K + 128);
  for (int i = 0; i < M; i++)
    for (int q = 0; q < K; q++)
      a[(size_t)i * K + q] = (int8_t)(((i * 13 + q * 7) % 31) - 15);
  for (int j = 0; j < N; j++)
    for (int q = 0; q < K; q++)
      b[(size_t)j * K + q] = (int8_t)(((j * 11 + q * 5) % 29) - 14);
  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++)
      c[(size_t)i * N + j] = (i * 3 + j) % 7 - 3;
  MemRef2 A = make_2(a, M, K), B = make_2(b, K, N), C = make_2(c, M, N);
  B.strides[0] = 1;
  B.strides[1] = K;
#if !defined(HOST_TEST)
  ame_fence();
#endif
  uint64_t start = nr_cycles();
  _mlir_ciface_kernel_matmul_i8_128x1024x128(&A, &B, &C);
  nr_puts("cycles matmul_i8_128x1024x128: ");
  nr_hex64(nr_cycles() - start);
  nr_puts("\r\n");
  unsigned errors = 0;
  int32_t max_error = 0;
  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++) {
      int32_t want = (i * 3 + j) % 7 - 3;
      for (int q = 0; q < K; q++)
        want += (int32_t)a[(size_t)i * K + q] * (int32_t)b[(size_t)j * K + q];
      int32_t got = c[(size_t)i * N + j], dlt = got - want;
      if (dlt < 0)
        dlt = -dlt;
      if (dlt > max_error)
        max_error = dlt;
      if (got != want) {
        if (errors < 4) {
          nr_puts("mismatch ");
          nr_hex32(i);
          nr_puts(" ");
          nr_hex32(j);
          nr_puts("\r\n");
        }
        errors++;
      }
    }
  return print_check("matmul_i8_128x1024x128", errors, (float)max_error);
}
#if defined(HOST_TEST)
/* Host builds need a process entry point.  On NR there is no main: crt.S
 * starts the runtime, which calls launch() (common/nr/nr_runtime.c). */
int main(void) { return launch(); }
#endif
