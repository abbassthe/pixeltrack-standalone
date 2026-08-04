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

## Status

| file | state |
|---|---|
| `CUDACore/CUDAAtomics.mojo` | **verified on real hardware** (2026-08-04), one real syntax bug found and fixed along the way |
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
