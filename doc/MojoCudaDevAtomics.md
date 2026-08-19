# Re-atomicising the CUDACore primitives

Working notes for §C7 of [MojoCudaDevPort.md](MojoCudaDevPort.md). Kept out of the
`.mojo` sources deliberately — the source files stay clean, the caveats live here.

## The problem

`serial` removed device concurrency; `mojo-serial` faithfully ported the removal. So
its `CUDACore` primitives look copyable but have no atomics at all:

| file | device-concurrency refs in `cudadev` | in `mojo-serial` |
|---|---|---|
| `prefixScan.h` | 25 | 0 |
| `SimpleVector.h` | 8 | 0 |
| `VecArray.h` | 4 | 0 |
| `AtomicPairCounter.h` | 1 | 0 |

Copying them into a GPU port compiles cleanly and races silently.

## Method

Take the `mojo-serial` file as the base — it is already working, reviewed Mojo with
the right structure, generics and naming — then add the atomicity back in against the
`cudadev` header. Do not re-translate from C++; that throws away work and invents
gratuitous differences from the sibling port.

## Do not port `mojo-serial/CUDACore/CUDACompat.mojo`

Its `struct CUDACompat` advertises `atomicAdd`, `atomicCAS`, `atomicMin`… but every
body is a plain non-atomic read-modify-write:

```mojo
fn atomicAdd(a, b):
    var ret = a[]
    a[] += b        # not atomic
    return ret
```

Its own `@deprecated` note says callers should be redirected to real operations. Here
the redirect target is `CUDACore/CUDAAtomics.mojo`.

Separately, that path collides with `mojocudatest`'s `CUDACompat.mojo`, which is the
actual CUDA runtime bindings. Both also define `CUDAStreamType` incompatibly
(`OpaquePointer` alias vs struct). `mojocudatest`'s file keeps the path.

## `CUDAAtomics.mojo` — what must be verified

All the unverified API surface is deliberately confined to that one file, so if the
guess is wrong there is one place to fix rather than four.

The call shape comes from a working call site,
`mojo-serial/MojoSerial/plugin_PixelTriplets/CAHitNtupletGeneratorKernels.mojo:200`:

```mojo
Atomic.fetch_add[ordering = Consistency.SEQUENTIAL](
    UnsafePointer(to=c.nEvents), UInt64(1)
)
```

**Resolved 2026-08-04, on real hardware** (RTX 4060, once a GPU became available — see
the GPU driver milestone in the main plan doc). Test: a kernel where 100,000 threads
each call `atomic_fetch_add(counter, 1)` concurrently and record their own returned
previous value into a per-thread array, followed by a second kernel doing 100,000
concurrent `atomic_fetch_sub(counter, 1)` back down. All three confirmed:

**1. Does the pointer form return the previous value?** Yes — the 100,000 recorded
previous values form an exact permutation of `0..99999`, i.e. every concurrent add
really did observe and return its own distinct pre-add value, not `None`/garbage/a
duplicate. No compare-exchange rewrite needed.

**2. Does `os.atomic.Atomic` lower to a real device atomic inside a GPU kernel?** Yes
— the final counter after 100,000 unsynchronized concurrent adds is exactly `100000`.
If the host `Atomic` type didn't lower to a true device atomic here, concurrent
non-atomic read-modify-writes would have lost updates and the result would be lower.

**3. Does `atomic_fetch_sub`'s add-of-negation trap on an unsigned dtype?** No — 100,000
concurrent subtracts from a starting value of 100,000 land exactly on `0`, no trap, no
corruption.

One real bug found in the process, unrelated to the above: `CUDAAtomics.mojo`'s
`ptr: UnsafePointer[Scalar[dt], mut=True]` parameter syntax doesn't parse in this
dialect (`inferred parameter passed out of order: 'mut'`) — this file had literally
never been compiled before (matches its own "written, unverified" status). Fixed to
`UnsafePointer[Scalar[dt], MutAnyOrigin]`, the idiom used everywhere else in this port.

Every primitive built on this file can now be trusted — verified, not just written.

## `AtomicPairCounter.add()` — the one place the pointer is not obvious

The other two primitives atomic-add to a plain `Int32` field, so
`UnsafePointer(to=self.m_size)` is direct. `AtomicPairCounter` is not: `cudadev` does
the atomic on a `union Atomic2`'s 64-bit `ac` member, while `mojo-serial` stores the
pair as `SIMD[DType.uint32, 2]` and derives the 64-bit view through `get_ac()` /
`set_ac()`. The atomic has to act on that 64-bit view, so the port reinterprets the
pointer:

```mojo
var previous = atomic_fetch_add(
    UnsafePointer(to=self.counter.c).bitcast[Scalar[DType.uint64]](), c
)
return Counters(bitcast[DType.uint32, 2](previous))
```

The `get_ac`/`set_ac` *logic* is not what changed — returning the pre-add value is
exactly what the original did (`var ret = self.counter` before the write) and what
`cudadev` does (`ret.ac = atomicAdd(&counter.ac, c)`). `OneToManyAssoc.h:233`
(`auto c = apc.add(n)`) depends on getting the previous value, so this must stay.
What changed is that the read and the write are now one indivisible operation instead
of two.

Three things to check on hardware, beyond the generic ones above:

- **Alignment.** A 64-bit atomic requires 8-byte alignment. `SIMD[DType.uint32, 2]`
  should be naturally 8-aligned, but a misaligned 64-bit atomic is a hard fault on
  CUDA, not a silent slowdown — confirm it.
- **`UnsafePointer.bitcast[T]()` spelling.** Taken to mean "new pointee type". The
  value-level `bitcast[DType, width]` used elsewhere in the file is a different
  function with a different parameter shape; do not assume one from the other.
- **Interior pointer into a `@register_passable("trivial")` struct.** `Counters` and
  `AtomicPairCounter` are both register-passable, so an interior pointer could in
  principle address a copy. In practice every call site holds the counter behind
  `UnsafePointer[AtomicPairCounter]` or `mut`/`ref` (see `HistoContainer.mojo:241`,
  `GPUCACell.mojo:413`, `CAHitNtupletGeneratorKernels.mojo:184`), so it does live in
  memory — but `mut self` on a register-passable type is worth confirming.

A cleaner alternative, if the bitcast proves troublesome: make `UInt64` the stored
field of `Counters` and derive the `SIMD[uint32, 2]` view instead. That gives a
direct, correctly-typed, aligned pointer with no bitcast. It was not done here
because it changes the struct's representation rather than just adding atomicity, and
it touches every `[0]`/`[1]` accessor in `HistoContainer` and `GPUCACell`.

### `atomic_fetch_min` — added for `gpuClustering.mojo`'s DSU, verified on hardware (2026-08-19)

`gpuClustering.h`'s `findClus` union-find loop needs `atomicMin(address, val)` returning the pre-update
value (to detect whether anything changed, and to symmetrically pull both union endpoints down). Checked
this Mojo version's actual API surface (via `strings` on the compiled `std.mojopkg`, since there's no
source to grep) rather than assuming: no `fetch_min`/`atomic_min` exists — only a non-returning
`Atomic.min[ordering](ptr, val)`. `gpu.primitives.block.min`/`warp.min` exist but solve a different
problem (many threads collapsing to one shared minimum) — not applicable here, since each thread updates
a different address (`clusterId[m]` for its own `m`).

Built from `Atomic.compare_exchange`, confirmed via spike to update its `expected` argument in place to
the real current value on failure (standard CAS ABI) — a plain CAS retry loop. Verified two ways: a
single-call correctness spike, and a 100,000-thread concurrent stress test (matching `fetch_add`'s own
verification scale below) where every thread submits a distinct value in `[0, 99999]` against a shared
location seeded above that range — final value landed on the true minimum (`0`), every returned "old"
value stayed within the valid range (ruling out torn reads), and exactly one thread observed the
untouched initial sentinel (as expected — only one CAS can be first against it).

### `syncthreads_or` — added for `gpuClustering.mojo`'s DSU, verified on hardware (2026-08-19)

`findClus`'s union-find loop is `while (__syncthreads_or(more))` — a single hardware instruction in
CUDA that barriers the block and OR-reduces `more` across every thread, atomically as one op. No direct
Mojo equivalent: `_barrier_and` exists as a symbol (found via `strings`) but isn't importable from
`std.gpu.sync` (where `barrier()` lives) without more setup than was worth chasing down, and there's no
`_barrier_or` at all.

Built instead from primitives already verified elsewhere in this port: `barrier()` (`PrefixScan.mojo`)
plus `atomic_fetch_add` on a shared counter (reset, add, read). `atomic_fetch_add`/`atomic_fetch_sub`/
`atomic_fetch_min` gained an `address_space` generic parameter (defaulting to `AddressSpace.GENERIC`, so
every existing call site is unaffected) to allow this shared-memory use — confirmed via `pixi run verify`
that `OneToManyAssoc.mojo`/`HistoContainer.mojo`'s existing calls still compile unchanged.

**Real bug caught before shipping**: the first version used two barriers (reset→barrier→add→barrier→read),
which is enough for a single call but not for a `while` loop calling this every iteration — nothing
stops a fast thread from finishing its read and resetting the shared counter for the *next* iteration
before a slower thread (in a different warp — warps don't run in lockstep with each other) has read the
*current* iteration's value. Added a third barrier after the read, closing the gap. A single-call test
wouldn't have caught this; verified instead with a 1024-thread block calling `syncthreads_or` 2000 times
in a tight loop with a rotating single-true-thread pattern each iteration (every iteration must return
true) — all 2000 correct with the fix, none would have been guaranteed correct without it.

`CUDACore/CUDASync.mojo` — new file (block-sync primitives are their own category, not atomics).

## Status

| file | state |
|---|---|
| `CUDACore/CUDAAtomics.mojo` | **verified on real hardware** (2026-08-04), one real syntax bug found and fixed along the way; `atomic_fetch_min` added and verified 2026-08-19 |
| `CUDACore/CUDASync.mojo` | `syncthreads_or` — **verified on real hardware** (2026-08-19), one real cross-iteration race found and fixed along the way |
| `CUDACore/AtomicPairCounter.mojo` | re-atomicised (1 site); underlying primitive now verified, `.add()` itself not yet directly hardware-tested |
| `CUDACore/SimpleVector.mojo` | re-atomicised (6 sites); not directly hardware-tested |
| `CUDACore/VecArray.mojo` | re-atomicised (4 sites); not directly hardware-tested |
| `CUDACore/PrefixScan.mojo` | **done, verified on hardware** (2026-08-04) — see below |

### `prefixScan` — done, and a real deadlock found along the way

Needed more than atomics: `__shfl_up_sync` (warp shuffle), `__ballot_sync` (warp
vote), `__syncthreads` (block barrier), and shared memory. None of those had a
Mojo counterpart anywhere in this port before now. Mapped 1:1 to
`std.gpu.primitives.warp.{shuffle_up,vote}`, `std.gpu.sync.barrier`,
`memory.stack_allocation[..., address_space=AddressSpace.SHARED]`. Device/host
split done via `sys.info.is_gpu()` (the Mojo spelling of `#ifdef __CUDA_ARCH__`).

`blockPrefixScan(off.data(), totOnes(), ws)` also uses `gpu.primitives.block.prefix_sum`
+ `broadcast` as an alternative building block for parts of the algorithm — a
ready-made, single-thread-per-value block scan the Mojo stdlib already ships, found
while scoping this.

**Real deadlock, found and root-caused on hardware.** C++'s `blockPrefixScan`
handles `size` not being a multiple of the block's thread count by having some
lanes exit their per-thread loop early (a divergent subset of a warp), then calling
masked `__shfl_up_sync(mask, ...)`/`__ballot_sync(mask, ...)` — completely standard,
well-defined CUDA. The direct Mojo translation of that — a subset of a warp calling
`vote()`/masked `shuffle_up` while the rest have branched away, even with a mask
that correctly matches exactly the executing lanes — hangs `ctx.synchronize()`
indefinitely in this Mojo/MAX version. Isolated with a minimal ~15-line repro
(unrelated to this file's own complexity) before escalating; a doc-equipped model
confirmed it's a known area of active bugs in Mojo's NVIDIA warp-shuffle lowering
(citing modular/modular#6799/#6800, a 64-bit-index-type shuffle splitting bug) and
that the documented/intended pattern is uniform participation: pad out-of-range
lanes with a neutral value and have every lane call the shuffle instruction every
time, gating only the *read/write of results* rather than the instruction itself —
exactly how `gpu.primitives.block.prefix_sum` apparently does it internally.

Rewrote on that basis: every lane always calls `shuffle_up`/`warpPrefixScan`
uniformly each round (padded with `0` when out of range), with `i < size` guards
only around reading `ci[i]` and writing `co[i]`/`ws[warpId]`. Verified correct
across 12 cases spanning every relevant boundary (`size` below/at/above a warp,
below/at/above `block_size`, multi-round) on both the device and host
(`is_gpu()`-false) paths — no hangs, all numerically correct.

Initially collapsed C++'s two `warpPrefixScan` overloads (`(ci,co,i,mask)` and the
in-place `(c,i,mask)`) into one value-in/value-out helper — flagged as an
unnecessary shape change from C++ with no real justification, unlike the loop
restructuring above (which *was* necessary for the divergence fix). Restructured
back to two array-indexed overloads matching C++ exactly, replacing the now-unusable
`mask` parameter with a `bound` (the `i < bound` check moves inside the function,
since padding replaces masking as the safety mechanism). Required parametrizing
over `AddressSpace` too, since `warpPrefixScan(ws, ...)` needs to accept the
`stack_allocation`-produced shared-memory pointer, not just generic-address-space
ones. Re-verified: same 12/12 passing.

Also ported the 4 C++ `assert`s: `0 == blockDim.x % 32` became a compile-time
`constrained[block_size % 32 == 0, ...]()` (stronger than C++'s runtime check,
since `block_size` is compile-time here); `size <= 1024` and `warpId < 32` became
`debug_assert`s. `assert(ws)` wasn't ported — C++'s `ws` is a caller-supplied
pointer that could be null by mistake; this port's `ws` is allocated internally
via `stack_allocation` right before use, so it can never be null by construction.

Pushed back on the block-uniform round-count loop (`for round in range(num_rounds)`)
above — asked whether C++'s condition-based loop shape (`while i < size`) could be
kept without the deadlock. It can, with one change: check each *warp's* base index
(`first - laneId`, uniform across all 32 lanes of a warp) instead of each thread's
own `i`. All 32 lanes of a warp then always enter/exit their loop together — no
within-warp divergence — while different warps can still stop after different
numbers of rounds, since warp-to-warp divergence is completely fine (shuffle only
operates within a single warp). This is strictly better than the round-count
version: closer to C++'s actual shape, and warps with zero relevant elements skip
the loop entirely instead of doing a wasted padding round. Re-verified across the
same boundary cases plus a many-empty-warps case (`size=17`, `block_size=256`, 7 of
8 warps entirely irrelevant) — correct, no hang.

## Bug found in `mojo-serial` while porting

`mojo-serial/CUDACore/SimpleVector.mojo:54` — `extend()` decrements by `1` on the
failure path where `cudadev/CUDACore/SimpleVector.h:89` undoes the full `size`:

```mojo
fn extend(mut self, size: Int32 = 1) -> Int32:
    ...
    else:
        self.m_size -= 1      # should be -= size
```

Fixed in this port. Worth fixing upstream in `mojo-serial` too — it is a live bug
there, independent of atomics.
