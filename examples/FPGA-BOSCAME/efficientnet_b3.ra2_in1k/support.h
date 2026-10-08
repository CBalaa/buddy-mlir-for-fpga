//===- support.h - EfficientNet operator host/board helpers ------*- C -*-===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// Shared by per-operator launch.c (host check and NR image). MemRef helpers
// match the MLIR C ABI. workspace() points into the 640 MiB NOLOAD arena
// defined in support.c (symbol effnet_arena).
//
// The full-model image must NOT link support.c: that object carries the
// 640 MiB arena. The model launch allocates its own HIGH workspace and
// labelled weight arena, same contract as qwen3-0.6b.
//
//===----------------------------------------------------------------------===//

#ifndef EFFNET_OPERATOR_SUPPORT_H
#define EFFNET_OPERATOR_SUPPORT_H
#include <stddef.h>
#include <stdint.h>
#ifdef HOST_TEST
#include <math.h>
#else
#include "nr_runtime.h"
#endif
#define MEMREF(R) typedef struct { void *allocated, *aligned; int64_t offset; int64_t sizes[R], strides[R]; } MemRef##R
MEMREF(1); MEMREF(2); MEMREF(3); MEMREF(4);
static inline MemRef1 make_1(void *p, int64_t a) { return (MemRef1){p,p,0,{a},{1}}; }
static inline MemRef2 make_2(void *p, int64_t a,int64_t b) { return (MemRef2){p,p,0,{a,b},{b,1}}; }
static inline MemRef3 make_3(void *p, int64_t a,int64_t b,int64_t c) { return (MemRef3){p,p,0,{a,b,c},{b*c,c,1}}; }
static inline MemRef4 make_4(void *p, int64_t a,int64_t b,int64_t c,int64_t d) { return (MemRef4){p,p,0,{a,b,c,d},{b*c*d,c*d,d,1}}; }
void *workspace(size_t byte_offset);
void nr_copy_bytes(void *destination, const void *source, size_t bytes);
int check_close(float actual, float expected, float atol, float rtol);
int print_check(const char *name, unsigned errors, float max_error);
void nr_puts(const char *s);
void nr_hex32(uint32_t x);
void nr_hex64(uint64_t x);
uint64_t nr_cycles(void);
#endif
