"""Cross-module checks for the NR memory support split out of nr_runtime.c.

The production nr_memory.c is compiled as its own translation unit with the
platform entries it calls (nr_puts/nr_hex64/nr_finish/abort) and the linker
symbols __heap_start/__heap_end supplied by the harness, so the test exercises
the real module boundary: the heap state and the memory/memrefCopy ABI live in
nr_memory.c, the fatal reporting path stays in nr_runtime.c.

The allocator checks pin the NR-specific contract that must not be unified with
the bare backend: 64-byte bump allocation, free() that does not reclaim, a
mark/reset that invalidates later allocations, and heap exhaustion reported as
FAIL through nr_finish.

Run with:  python3 -m unittest discover examples/FPGA-BOSCAME/tools
"""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


FPGA = Path(__file__).resolve().parents[1]
NR = FPGA / "common" / "nr"
HEAP_BYTES = 4096

HARNESS = r'''
#include <assert.h>
#include <inttypes.h>
#include <setjmp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- platform entries normally provided by nr_runtime.c ---- */
static jmp_buf fatal_jump;
static int fatal_status;
static int fatal_count;
static char report[1024];
static size_t report_length;

void nr_puts(const char *text) {
  size_t length = strlen(text);
  assert(report_length + length < sizeof(report));
  memcpy(report + report_length, text, length);
  report_length += length;
  report[report_length] = 0;
}
void nr_hex64(uint64_t value) {
  char text[17];
  snprintf(text, sizeof(text), "%016" PRIx64, value);
  nr_puts(text);
}
__attribute__((noreturn)) void nr_finish(int status) {
  fatal_status = status;
  ++fatal_count;
  longjmp(fatal_jump, 1);
}
/* nr_memory.c's abort() is renamed to nr_test_abort so the harness keeps the
 * host libc's abort() for assert() failures; a real assert must be visible as
 * SIGABRT instead of being swallowed by the test's own non-returning hook. */
__attribute__((noreturn)) void nr_test_abort(void) {
  fatal_status = -2;
  ++fatal_count;
  longjmp(fatal_jump, 1);
}

/* ---- linker-owned heap range used by the module under test ---- */
static unsigned char heap_area[__NR_HEAP_BYTES__] __attribute__((aligned(64)));
__asm__(".globl __heap_start\n"
        "__heap_start = heap_area\n"
        ".globl __heap_end\n"
        "__heap_end = heap_area + __NR_HEAP_BYTES__\n");

/* nr_memory.c is compiled with these entry points renamed, so the harness can
 * keep using the host libc for its own bookkeeping. */
void *nr_test_malloc(size_t size);
void *nr_test_calloc(size_t count, size_t size);
void *nr_test_aligned_alloc(size_t alignment, size_t size);
void nr_test_free(void *pointer);
void *nr_test_memcpy(void *dst, const void *src, size_t count);
void *nr_test_memmove(void *dst, const void *src, size_t count);
void *nr_test_memset(void *dst, int value, size_t count);
int nr_test_memcmp(const void *lhs, const void *rhs, size_t count);
void nr_test_memrefCopy(int64_t element_size, void *source, void *destination);
__attribute__((noreturn)) void nr_test_abort(void);

uintptr_t nr_heap_mark(void);
void nr_heap_reset(uintptr_t mark);

/* ---- MLIR CRunner descriptor layout, as the lowered kernels pass it ---- */
typedef struct {
  int64_t rank;
  void *descriptor;
} UnrankedMemRef;
typedef struct {
  char *base;
  char *data;
  int64_t offset;
  int64_t dimensions[];
} RankedMemRef;

static void reset_fatal(void) {
  fatal_status = 12345;
  report_length = 0;
  report[0] = 0;
}

static int fail_if_no_fatal(void) { return fatal_count == 0; }

static void test_heap_bump_is_64_byte_aligned(void) {
  uintptr_t start = (uintptr_t)heap_area;
  void *first, *second;
  assert(nr_heap_mark() == start);
  /* No allocation is returned before the linker-owned start. */
  first = nr_test_malloc(1);
  assert(first != NULL);
  assert((uintptr_t)first >= start);
  assert(((uintptr_t)first % 64) == 0);
  /* 1 byte is rounded up to the 64-byte quantum, so the next block is +64. */
  second = nr_test_malloc(64);
  assert((uintptr_t)second == (uintptr_t)first + 64);
  assert(((uintptr_t)second % 64) == 0);
}

static void test_free_does_not_reclaim(void) {
  void *first, *second, *third;
  nr_heap_reset((uintptr_t)heap_area);
  first = nr_test_malloc(64);
  second = nr_test_malloc(64);
  nr_test_free(first);            /* NR free() is a no-op */
  nr_test_free(NULL);             /* and tolerates NULL */
  third = nr_test_malloc(64);
  assert(third == (void *)((uintptr_t)second + 64));
  assert(third != first);
}

static void test_heap_mark_and_reset(void) {
  uintptr_t mark;
  void *first, *again;
  nr_heap_reset((uintptr_t)heap_area);
  first = nr_test_malloc(128);
  mark = nr_heap_mark();
  assert(mark == (uintptr_t)first + 128);
  (void)nr_test_malloc(256);
  assert(nr_heap_mark() == mark + 256);
  nr_heap_reset(mark);
  assert(nr_heap_mark() == mark);
  again = nr_test_malloc(128);
  assert((uintptr_t)again == mark);   /* the reset mark is the next block */
}

static void test_invalid_heap_mark_reports_fail(void) {
  reset_fatal();
  if (setjmp(fatal_jump) == 0) {
    nr_heap_reset((uintptr_t)heap_area - 64);
    assert(fail_if_no_fatal());
  }
  assert(fatal_status == 1);
  assert(strstr(report, "invalid heap mark") != NULL);
}

static void test_heap_exhaustion_reports_size_and_limit(void) {
  reset_fatal();
  if (setjmp(fatal_jump) == 0) {
    for (unsigned i = 0; i < 1000; ++i) {
      void *block = nr_test_malloc(256);
      assert(block != NULL);
      nr_test_memset(block, i, 256);
    }
    assert(fail_if_no_fatal());
  }
  assert(fatal_status == 1);
  assert(strstr(report, "heap exhausted") != NULL);
}

static void test_calloc_zeroes_and_checks_overflow(void) {
  unsigned char *block;
  void *overflow;
  nr_heap_reset((uintptr_t)heap_area);
  block = nr_test_calloc(16, 8);
  assert(block != NULL);
  for (unsigned i = 0; i < 128; ++i) assert(block[i] == 0);
  /* The product check must reject an overflowing request without touching the
   * heap: NR calloc returns NULL, it does not report exhaustion. */
  overflow = nr_test_calloc(SIZE_MAX / 2 + 1, 4);
  assert(overflow == NULL);
}

static void test_aligned_alloc_rejects_bad_alignment(void) {
  void *block;
  nr_heap_reset((uintptr_t)heap_area);
  assert(nr_test_aligned_alloc(0, 16) == NULL);
  assert(nr_test_aligned_alloc(3, 16) == NULL);
  block = nr_test_aligned_alloc(128, 16);
  assert(block != NULL);
  assert(((uintptr_t)block % 128) == 0);
}

static void test_scalar_memory_helpers(void) {
  unsigned char source[37], destination[37];
  char overlap[32];
  for (unsigned i = 0; i < sizeof(source); ++i) source[i] = (unsigned char)(i * 7 + 1);
  nr_test_memset(destination, 0xa5, sizeof(destination));
  assert(nr_test_memcmp(destination, source, sizeof(source)) != 0);
  assert(nr_test_memcpy(destination, source, sizeof(source)) == destination);
  assert(nr_test_memcmp(destination, source, sizeof(source)) == 0);
  /* Unaligned endpoints and an odd trailing byte. */
  nr_test_memcpy(destination + 1, source + 1, 35);
  assert(nr_test_memcmp(destination + 1, source + 1, 35) == 0);
  nr_test_memset(destination, 0x5a, 37);
  for (unsigned i = 0; i < 37; ++i) assert(destination[i] == 0x5a);
  for (unsigned i = 0; i < 32; ++i) overlap[i] = (char)i;
  nr_test_memmove(overlap + 8, overlap, 24);      /* forward overlap */
  for (unsigned i = 0; i < 24; ++i) assert(overlap[i + 8] == (char)i);
  nr_test_memmove(overlap, overlap + 8, 24);      /* backward overlap */
  for (unsigned i = 0; i < 24; ++i) assert(overlap[i] == (char)i);
}

struct descriptors {
  UnrankedMemRef unranked;
  RankedMemRef ranked;
  int64_t dimensions[16];
};

static void init_descriptor(struct descriptors *descriptor, char *data,
                            int64_t rank, const int64_t *sizes,
                            const int64_t *strides, int64_t offset) {
  descriptor->ranked.base = data;
  descriptor->ranked.data = data;
  descriptor->ranked.offset = offset;
  for (int64_t i = 0; i < rank; ++i) {
    descriptor->dimensions[i] = sizes[i];
    descriptor->dimensions[rank + i] = strides[i];
  }
  descriptor->unranked.rank = rank;
  descriptor->unranked.descriptor = &descriptor->ranked;
}

static void test_memref_copy_contiguous_and_strided(void) {
  int32_t source[4][3] = {{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}};
  int32_t offset_source[5] = {10, 11, 12, 13, 14};
  int32_t offset_target[5] = {0, 0, 0, 0, 0};
  int32_t flat[12], transposed[12];
  int64_t sizes[2] = {4, 3};
  int64_t contiguous_strides[2] = {3, 1};
  int64_t transpose_strides[2] = {1, 4};
  int64_t one_size[1] = {3};
  int64_t one_stride[1] = {1};
  struct descriptors src, dst;

  for (unsigned i = 0; i < 12; ++i) { flat[i] = -1; transposed[i] = -1; }
  init_descriptor(&src, (char *)source, 2, sizes, contiguous_strides, 0);
  init_descriptor(&dst, (char *)flat, 2, sizes, contiguous_strides, 0);
  nr_test_memrefCopy((int64_t)sizeof(int32_t), &src.unranked, &dst.unranked);
  for (unsigned i = 0; i < 12; ++i) assert(flat[i] == source[i / 3][i % 3]);

  /* A transposed destination takes the strided path: element (row, col) is
   * stored at row * 1 + col * 4, so every element lands where it belongs. */
  init_descriptor(&dst, (char *)transposed, 2, sizes,
                  transpose_strides, 0);
  nr_test_memrefCopy((int64_t)sizeof(int32_t), &src.unranked, &dst.unranked);
  for (unsigned row = 0; row < 4; ++row)
    for (unsigned col = 0; col < 3; ++col)
      assert(transposed[row + col * 4] == source[row][col]);

  /* A non-zero offset addresses from base + offset * element_size; the
   * elements outside the descriptor stay untouched. */
  init_descriptor(&src, (char *)offset_source, 1, one_size, one_stride, 1);
  init_descriptor(&dst, (char *)offset_target, 1, one_size, one_stride, 1);
  nr_test_memrefCopy((int64_t)sizeof(int32_t), &src.unranked, &dst.unranked);
  assert(offset_target[0] == 0 && offset_target[4] == 0);
  assert(offset_target[1] == 11 && offset_target[2] == 12);
  assert(offset_target[3] == 13);

  /* Mismatched shapes are rejected without touching the destination. */
  reset_fatal();
  init_descriptor(&src, (char *)source, 2, sizes, contiguous_strides, 0);
  sizes[1] = 2;
  init_descriptor(&dst, (char *)flat, 2, sizes, contiguous_strides, 0);
  if (setjmp(fatal_jump) == 0) {
    nr_test_memrefCopy((int64_t)sizeof(int32_t), &src.unranked, &dst.unranked);
    assert(fail_if_no_fatal());
  }
  assert(fatal_status == -2);       /* shape mismatch takes the abort() path */
}

static void test_memref_copy_rejects_bad_rank(void) {
  int64_t sizes[1] = {4};
  int64_t strides[1] = {1};
  int32_t storage[4] = {0, 0, 0, 0};
  struct descriptors src, dst;
  int before = fatal_count;
  reset_fatal();
  init_descriptor(&src, (char *)storage, 1, sizes, strides, 0);
  init_descriptor(&dst, (char *)storage, 1, sizes, strides, 0);
  src.unranked.rank = 9;                 /* above the rank <= 8 contract */
  if (setjmp(fatal_jump) == 0) {
    nr_test_memrefCopy((int64_t)sizeof(int32_t), &src.unranked, &dst.unranked);
    assert(fail_if_no_fatal());
  }
  assert(fatal_status == -2);            /* abort() path, not nr_finish() */
  assert(fatal_count == before + 1);
}

static void run(const char *name, void (*test)(void)) {
  int before = fatal_count;
  /* Defensive jump target: a fatal that escapes a test must not resume in a
   * dead frame.  Tests that expect a fatal install their own setjmp. */
  if (setjmp(fatal_jump) == 0)
    test();
  assert(fatal_count - before <= 1);
  printf("  ok %s\n", name);
}

int main(void) {
  reset_fatal();
  printf("nr_memory: heap range %p..%p\n", (void *)heap_area,
         (void *)(heap_area + __NR_HEAP_BYTES__));
  run("heap bump is 64-byte aligned", test_heap_bump_is_64_byte_aligned);
  run("free does not reclaim", test_free_does_not_reclaim);
  run("heap mark and reset", test_heap_mark_and_reset);
  run("invalid heap mark reports fail", test_invalid_heap_mark_reports_fail);
  run("heap exhaustion reports fail", test_heap_exhaustion_reports_size_and_limit);
  run("calloc zeroes and checks overflow", test_calloc_zeroes_and_checks_overflow);
  run("aligned_alloc rejects bad alignment",
      test_aligned_alloc_rejects_bad_alignment);
  run("scalar memory helpers", test_scalar_memory_helpers);
  run("memrefCopy contiguous and strided",
      test_memref_copy_contiguous_and_strided);
  run("memrefCopy rejects bad rank", test_memref_copy_rejects_bad_rank);
  printf("nr_memory: all checks passed\n");
  return 0;
}
'''.replace("__NR_HEAP_BYTES__", str(HEAP_BYTES))

RENAMES = {
    "malloc": "nr_test_malloc",
    "calloc": "nr_test_calloc",
    "aligned_alloc": "nr_test_aligned_alloc",
    "free": "nr_test_free",
    "memcpy": "nr_test_memcpy",
    "memmove": "nr_test_memmove",
    "memset": "nr_test_memset",
    "memcmp": "nr_test_memcmp",
    "memrefCopy": "nr_test_memrefCopy",
    "abort": "nr_test_abort",
}


@unittest.skipUnless(shutil.which("cc"), "a host C compiler is required")
class NrMemoryModuleTest(unittest.TestCase):
    def build_and_run(self, directory, extra_flags=()):
        harness = Path(directory) / "harness.c"
        harness.write_text(HARNESS)
        harness_object = Path(directory) / "harness.o"
        module_object = Path(directory) / "nr_memory.o"
        check = Path(directory) / "check"
        base = ["cc", "-std=c11", "-O1", "-g", "-Wall", "-Wextra",
                "-Werror", f"-I{NR}", *extra_flags]
        subprocess.run([*base, "-c", str(harness), "-o", str(harness_object)],
                       check=True, capture_output=True, text=True)
        renames = [f"-D{name}={target}" for name, target in RENAMES.items()]
        subprocess.run([*base, *renames, "-c", str(NR / "nr_memory.c"),
                        "-o", str(module_object)],
                       check=True, capture_output=True, text=True)
        subprocess.run(["cc", *extra_flags, str(harness_object),
                        str(module_object), "-o", str(check)],
                       check=True, capture_output=True, text=True)
        return subprocess.run([str(check)], check=True, capture_output=True,
                              text=True).stdout

    def test_production_module_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            output = self.build_and_run(directory)
        self.assertIn("nr_memory: all checks passed", output)

    def test_optional_address_sanitizer_build(self):
        # The memory helpers must stay clean under ASan too; the module is
        # compiled into the same process, so the harness's own allocations are
        # not exercised here, only the module's address arithmetic.
        with tempfile.TemporaryDirectory() as directory:
            output = self.build_and_run(
                directory, ("-fsanitize=address,undefined",
                            "-fno-omit-frame-pointer"))
        self.assertIn("nr_memory: all checks passed", output)


if __name__ == "__main__":
    unittest.main()
