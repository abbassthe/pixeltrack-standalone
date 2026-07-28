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

Two things are unconfirmed. Both are checkable with a one-line test on a linux-64 +
NVIDIA host (§0 of the plan — this machine has neither).

**1. Does the static pointer form return the previous value?**
At that call site the result is discarded *without* `_ =`, which hints the static form
may return `None`. Every caller here needs the previous value — CUDA's semantics are
`auto previousSize = atomicAdd(&m_size, 1)`. The *instance* form definitely returns
the old value: `mojo-serial/plugin_PixelTriplets/BrokenLineFitOnGPU.mojo:72` relies on
`done.fetch_add(1) == 0`.

*If it returns `None`:* rewrite the two bodies in `CUDAAtomics.mojo` as a
compare-exchange loop. Nothing outside that file changes.

**2. Does `os.atomic.Atomic` lower to a real device atomic inside a GPU kernel?**
It is the host atomic type. If it does not lower correctly on device, swap the bodies
for the `gpu` module's atomic intrinsics.

**3. `atomic_fetch_sub` is expressed as an add of the negation** so it depends on only
the one primitive above. For unsigned `dt` this relies on wraparound, which matches
what CUDA's `atomicSub` already does — but confirm Mojo does not trap on it.

Until 1–3 are checked, every primitive built on this file is **written but unverified**.

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
| `CUDACore/CUDAAtomics.mojo` | written, unverified |
| `CUDACore/AtomicPairCounter.mojo` | re-atomicised (1 site) |
| `CUDACore/SimpleVector.mojo` | re-atomicised (6 sites) |
| `CUDACore/VecArray.mojo` | re-atomicised (4 sites) |
| `CUDACore/PrefixScan.mojo` | **not started** — see below |

### `prefixScan` is a separate, larger job

It needs more than atomics: `__shfl_up_sync` (warp shuffle), `__ballot_sync` (warp
vote), `__syncthreads` (block barrier), `__threadfence`, dynamic shared memory, and
`extern __shared__`. None of those have a counterpart in either donor port, and their
Mojo spellings are unverified. Treat it as its own piece of work alongside the
Phase 1 GPU primitives (`launch`, `copyAsync`), not as part of this pass.

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
