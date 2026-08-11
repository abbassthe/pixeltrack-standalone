# Porting `cudadev` to Mojo — Work Plan

**Target:** `src/mojo-cudadev/` — a Mojo port of `src/cudadev`, the development CUDA
version of the pixel-tracking reconstruction, running real GPU kernels via
`std.gpu.host.DeviceContext`.

**Status:** Phase 1 in progress. Real NVIDIA GPU now available and verified (2026-07-30) —
see the note at the end of §10. Everything before that note was written against a
GPU-less machine and is compile-verified only, per the original §0 caveat; everything
from that note onward is hardware-verified on real GPU execution.
**Date:** 2026-07-27

---

## 0. Blocker found before anything else: the two donor ports don't share a Mojo version

This was not visible from comparing C++ sources and it drives the whole schedule.

`serial/Framework` and `cudadev/Framework` are byte-identical C++ (§1b). But the two
*Mojo* ports of that same framework do **not** agree — `Event.mojo` differs by 72
lines, `PluginFactory.mojo` by 106, `ESPluginFactory.mojo` by 77, `EventSetup.mojo`
by 46. Inspecting the diffs, essentially none of it is semantics. It is **Mojo
language drift**:

| `mojo-serial` (older) | `mojocudatest` (newer) |
|---|---|
| `UnsafePointer[NoneType]` | `UnsafePointer[NoneType, MutAnyOrigin]` |
| `struct X(Copyable, Movable, …)` | `struct X(Copyable, ImplicitlyCopyable, Movable, …)` |
| `T: Typeable & Movable` | `T: Typeable & Movable & ImplicitlyDestructible` |
| `UnsafePointer[T].alloc(1)` | `alloc[Self.T](1)` |
| `__copyinit__(out self, other: Self)` | `__copyinit__(out self, copy: Self)` |
| `__moveinit__(out self, var other: Self)` | `__moveinit__(out self, var take: Self)` |
| bare `T` in parameter position | `Self.T` |

Measured occurrences:

```
                      MutAnyOrigin  ImplicitlyCopyable  ImplicitlyDestructible  .alloc(   UnsafePointer
mojo-serial                      0                   0                       0       11    598 / 62 files
mojocudatest              155 / 22f            20 / 11f                17 / 7f        0    199 / 31 files
```

Cause is visible in the manifests: `mojo-serial/pixi.toml` pins `mojo = "25.5.*"`,
while `mojocudatest/pixi.toml` floats `max = ">=25.3"` and has drifted forward.
`pixi search` currently offers `1.0.0b2` / `0.26.x` / `0.25.7` — **`25.5.*` is not
available for osx-arm64 at all**, and the version line has been renumbered upstream.

### Consequence

You cannot simply copy Bucket A files from both donors into one package — they will
not compile together. Phase 0 must first **choose one Mojo version and migrate the
other donor forward to it.**

**Standardise on the newer Mojo (mojocudatest's dialect).** Pinning to 25.5 to spare
`mojo-serial` is a dead end — GPU `DeviceContext` work needs the current toolchain,
and 25.5 is already unobtainable on some platforms.

### Conflict rule (decided)

> **When a file exists in both donor ports, take `mojocudatest`'s version.**

No per-file deliberation. `mojo-serial` supplies only what `mojocudatest` does not
have — which is every algorithm, and the six `MojoBridge` maths/utility modules.

**Scope limit — this rule is about donor ports, not about C++ lineage.** It resolves
`mojo-serial` vs `mojocudatest`. It does *not* override `cudadev`'s divergence from
`cuda`: where the C++ genuinely changed, `cudadev` wins even though `mojocudatest`
ported the older shape. The live case is §C2 — `mojocudatest` ported the notcub
allocators, but `cudadev` replaced them with `GenericCachingAllocator`, so those two
files still get rewritten, not kept.

**Documented exception — `CUDACore/CUDACompat.mojo`.** The two donors ship a file at
the same path whose contents are *disjoint*, so "take `mojocudatest`'s" would
silently delete something rather than pick a version:

| donor | what `CUDACompat.mojo` actually contains |
|---|---|
| `mojocudatest` | CUDA **runtime API bindings** — `CUDARuntime`, `cudaGetDevice`, `cudaSetDevice`, `cudaEventCreateWithFlags`, `cudaEventRecord`, … |
| `mojo-serial` | `struct CUDACompat` — CPU **emulations of device atomics** (`atomicCAS`, `atomicInc`, `atomicAdd`, `atomicSub`, `atomicMin`, `atomicMax`) |

They also both define `CUDAStreamType`, incompatibly: an `OpaquePointer` alias in
`mojo-serial`, a `struct` in `mojocudatest`.

Resolution: `mojocudatest`'s file keeps the path (the rule holds for the runtime
bindings). `mojo-serial`'s atomics do **not** get copied at all — see §C7.

The conflict set is exactly 26 files, all resolved to `mojocudatest` in Phase 0b:

| Dir | Files (diff lines vs `mojo-serial`) |
|---|---|
| `Framework/` | `PluginFactory` (106), `ESPluginFactory` (77), `Event` (72), `EventSetup` (46), `ProductRegistry` (17), `EDGetToken` (10), `EDPutToken` (10), `EDProducer` (8), `ESProducer` (6) |
| `bin/` | `Source` (113), `StreamSchedule` (102), `EventProcessor` (63), `PosixClockGettime` (10), `main` (see below) |
| `MojoBridge/` | `DTypes` (246), `OrderedMap` (58), `OrderedMultiSet` (34), `File` (12) |
| `DataFormats/` | `FEDRawData` (18), `FEDRawDataCollection` (18), `VertexCount` (13), `DigiClusterCount` (5), `TrackCount` (5), `FEDHeader` (2), `FEDTrailer` (2), `FEDNumbering` (1) |

Concrete resolutions this rule settles:
- `EDProducer.endJob()` — `mojocudatest`'s non-`raises` signature wins. Every plugin
  ported from `mojo-serial` must drop `raises` on `endJob`.
- `bin/main.mojo` — `mojocudatest`'s `main.mojo` is the base (`mojo-serial` has its
  own under `bin/`). Its `plugin-Test1` registrations were dropped; cudadev's
  producers get registered there in Phase 4.

Cost of migrating `mojo-serial` forward: the mechanical part is small and bounded —
11 `.alloc(` sites (all listed below), 8 `__copyinit__` and 25 `__moveinit__`
signature renames. The **unbounded** part is trait conformance: ~38 structs declare
`(Copyable, Defaultable, Movable, Typeable)` and may each need `ImplicitlyCopyable`,
and 598 `UnsafePointer` uses may or may not need explicit origins.

> **This is the single biggest unknown in the plan and it cannot be resolved by
> reading — it needs a compiler.** See the Phase 0 spike in §7.

The 11 `.alloc(` sites in `mojo-serial`:
```
bin/Source.mojo:242              Framework/Event.mojo:43
Framework/EventSetup.mojo:30     Framework/PluginFactory.mojo:33
Framework/ESPluginFactory.mojo:32   MojoBridge/Static.mojo:62
CUDADataFormats/HeterogeneousSoA.mojo:26,30,34,38,44
```

### Also: this machine cannot build or run the port

`uname -sm` → `Darwin arm64`; `mojo` is not installed; no NVIDIA GPU. Both donor
ports declare `platforms = ["linux-64"]`. All Phase 0–2 work below is textual and
can proceed here, but **nothing is compile-verified until it runs on a linux-64 +
NVIDIA host.** Treat every "done" in this document as "written, not verified" until
then.

---

## 1. What we are starting from

Two existing Mojo ports supply almost everything we need, from opposite directions:

| Port | Ported from | Size | Supplies |
|------|-------------|------|----------|
| [`src/mojo-serial/`](../src/mojo-serial/) | `serial` | ~22.0k lines, 157 files | Every **reconstruction algorithm** — clusterizer, RecHits, CA triplets, Riemann/BrokenLine fits, vertex finder — plus `Framework`, `DataFormats`, `Geometry`, `CondFormats`, and the `MojoBridge` support layer (`Matrix`, `SymmetricEigen`, `OrderedMap`, …). CPU only. |
| [`src/mojocudatest/`](../src/mojocudatest/) | `cudatest` | ~7.6k lines, 70 files | The entire **CUDACore GPU runtime layer** — `ScopedContext`, `Product`/`ProductBase`, `StreamCache`, `EventCache`, `ContextState`, caching device/host allocators, `device_unique_ptr`/`host_unique_ptr`, `chooseDevice`, `cudaCheck` — and a proven `DeviceContext` kernel launch in [`gpuAlgo1.mojo`](../src/mojocudatest/plugin-Test1/gpuAlgo1.mojo). |

`cudadev` non-test source is **22,291 lines across 205 files**. The job is
substantially *merging* the two existing ports and then closing the
`cuda` → `cudadev` delta, not writing 22k lines from scratch.

### The two decisive measurements

Everything below hangs off these, so they are stated up front.

**(a) `cudatest/CUDACore` is byte-identical to `cudadev/CUDACore` for 25 of 32 shared files.**
`diff -w` line counts, `cudatest` → `cudadev`:

```
ContextState.{cc,h}  EventCache.{cc,h}  Product.h  ProductBase.{cc,h}
ScopedContext.{cc,h} SharedEventPtr.h   SharedStreamPtr.h  StreamCache.{cc,h}
chooseDevice.{cc,h}  cudaCheck.h        currentDevice.h    deviceCount.h
eventWorkHasCompleted.h  allocate_host.h  host_unique_ptr.h
CachingDeviceAllocator.h  CachingHostAllocator.h              →  0 diff lines
```
The only non-zero ones are `getCachingHostAllocator.h` (100), `getCachingDeviceAllocator.h`
(96), `device_unique_ptr.h` (27), `ScopedSetDevice.h` (25), `allocate_device.cc` (16),
`allocate_host.cc` (7), `deviceAllocatorStatus.h` (7), `allocate_device.h` (6),
`deviceAllocatorStatus.cc` (2).

**So the GPU runtime layer is already ported.** That is the single biggest reason
this port is tractable.

**(b) `serial/Framework` is byte-identical to `cudadev/Framework` for all 24 shared files.**
Only `CMSUnrollLoop.h` and `propagate_const_array.h` are new. The framework
semantics did not change between the CPU and development-CUDA branches.

---

## 2. Bucket A — Reuse as-is (mechanical move, no logic work)

Copy the Mojo file over, fix the import path, done. No algorithmic reasoning required.

### A1. GPU runtime layer — from `mojocudatest`
25 files, ~2.5k lines of Mojo. Verified identical between `cudatest` and `cudadev`.

```
CUDACore/ContextState        CUDACore/EventCache          CUDACore/Product
CUDACore/ProductBase         CUDACore/ScopedContext       CUDACore/SharedEventPtr
CUDACore/SharedStreamPtr     CUDACore/StreamCache         CUDACore/chooseDevice
CUDACore/cudaCheck           CUDACore/currentDevice       CUDACore/deviceCount
CUDACore/eventWorkHasCompleted   CUDACore/host_unique_ptr
CUDACore/CachingDeviceAllocator  CUDACore/CachingHostAllocator
CUDACore/CUDACompat          CUDACore/CUDAAppContext
```

### A2. Framework — from `mojocudatest`
The C++ is identical between `serial` and `cudadev`, so either Mojo port is
semantically valid; the §0 conflict rule takes `mojocudatest`'s. It is also the only
one with the async pieces `cudadev` needs:

- Shared with `mojo-serial` (rule applies): `EDGetToken`, `EDProducer`,
  `EDPutToken`, `ESPluginFactory`, `ESProducer`, `Event`, `EventSetup`,
  `PluginFactory`, `ProductRegistry`
- Only in `mojocudatest`: `TaskBase`, `WaitingTask`, `WaitingTaskHolder`,
  `WaitingTaskWithArenaHolder`, `ReusableObjectHolder`

### A3. Algorithm/data files with **zero** `cuda`→`cudadev` change

**Already taken from `mojocudatest`** (in both donors, so the §0 rule applies) —
done in Phase 0b:
- `DataFormats/` — `FEDHeader`, `FEDTrailer`, `FEDNumbering`, `FEDRawData`,
  `FEDRawDataCollection`, `DigiClusterCount`, `TrackCount`, `VertexCount`
- `bin/` — `EventProcessor`, `Source`, `StreamSchedule`, `PosixClockGettime`, `main`
- `MojoBridge/` — `DTypes`, `File`, `OrderedMap`, `OrderedMultiSet`, `ConcurrentQueue`

**From `mojo-serial`** (no `mojocudatest` counterpart exists, so no conflict) —
pending the dialect migration:
- `DataFormats/approx_atan2.h` — 0 diff both `serial`→`cuda` and `cuda`→`cudadev`
- `DataFormats/SOARotation.h` — 0 diff both
- `DataFormats/BeamSpotPOD.h`, `SiPixelRawDataError.h`, `PixelErrors`, `SiPixelDigisSoA`
- `CUDACore/eigenSoA.h` — 0 diff `cuda`→`cudadev`
- `plugin-PixelVertexFinding/gpuClusterTracksByDensity.h` (8), `gpuSplitVertices.h` (6),
  `gpuSortByPt2.h` (~small), `gpuFitVertices.h`, `gpuClusterTracksDBSCAN/Iterative.h` —
  vertex finding barely moved between `cuda` and `cudadev`, but check each against
  §C7 before copying: these are kernels and several use `atomicAdd`/`__syncthreads`
- `Geometry/phase1PixelTopology.h`
- `MojoBridge/` — `Matrix`, `Vector`, `SymmetricEigen`, `Print`, `Static`, `Timer`

> `CUDACore/prefixScan.h`, `SimpleVector.h`, `VecArray.h` and `AtomicPairCounter.h`
> were originally listed here. They are **not** Bucket A — see §C7. The `cuda`→`cudadev`
> diff is small, but `mojo-serial` stripped their atomics, so the Mojo side is not a
> copy.

**Bucket A total: roughly 90–100 files.** The real cost here is import-path
normalisation (see §6), not logic.

---

## 3. Bucket B — Minimal change (mechanical, but touch every line)

Large `diff` counts that are **naming-convention churn**, not algorithm change.
`cudadev` adopted CMS code-style naming wholesale. Verified by inspection:

| File | `cuda`→`cudadev` diff lines | Nature of change |
|------|------|------|
| `plugin-PixelTriplets/BrokenLine.h` | 595 | `namespace BrokenLine` → `brokenline`; `Rfit::` → `riemannFit::`; `circle_fit` → `CircleFit`; template `<int N>` → `<int n>`; members `q`→`qCharge`, `s`→`sTransverse`, `S`→`sTotal`, `Z`→`zInSZplane`, `VarBeta`→`varBeta`; doc-comment reflow |
| `plugin-PixelTriplets/RiemannFit.h` | 557 | same rename sweep |
| `plugin-PixelTriplets/FitUtils.h` | 114 | same |
| `plugin-PixelTriplets/CAConstants.h` | 93 | `namespace CAConstants` → `caConstants`; `constexpr uint32_t maxNumberOfTuples()` **functions → plain constexpr values**; `#ifndef ONLY_PHICUT` inverted to `#ifdef` |
| `CondFormats/pixelCPEforGPU.h` | 51 | renames + small additions |
| `plugin-SiPixelClusterizer/gpuCalibPixel.h` | 17 | renames |
| `plugin-PixelTriplets/choleskyInversion.h` | 25 | renames |

**How to work these:** start from the `mojo-serial` file (already correct Mojo,
already validated), apply the rename table, adjust `CAConstants` call sites from
`maxNumberOfTuples()` to `maxNumberOfTuples`. Low risk, but do **not** assume it is
*only* renames — diff each one against `cuda` with the renames normalised away
before signing it off. Budget ~1 day for the whole PixelTriplets rename sweep.

Also in this bucket:
- `CUDADataFormats/ZVertexSoA.h` (8), `TrackingRecHit2DSOAView.h` (18),
  `plugin-PixelTriplets/gpuPixelDoubletsAlgos.h` (45), `gpuFishbone.h`,
  `plugin-SiPixelClusterizer/gpuClustering.h` (54), `gpuClusterChargeCut.h` (63),
  `plugin-SiPixelRecHits/gpuPixelRecHits.h` (49)
- The 7 non-identical CUDACore files from §1(a): `device_unique_ptr` (27),
  `ScopedSetDevice` (25), `allocate_device.{cc,h}` (22), `allocate_host.cc` (7),
  `deviceAllocatorStatus.{cc,h}` (9) — small, well-understood edits on top of
  the `mojocudatest` Mojo versions.

---

## 4. Bucket C — Real work (structural change, needs design)

### C1. `OneToManyAssoc` / `HistoContainer` inversion — **do this first**
In `cuda`/`serial`, `OneToManyAssoc` is a *type alias for* `HistoContainer`.
[`mojo-serial/.../HistoContainer.mojo:355`](../src/mojo-serial/MojoSerial/CUDACore/HistoContainer.mojo#L355)
encodes exactly that: `alias OneToManyAssoc[...]`.

In `cudadev` the relationship is **inverted and split into two files**:
`OneToManyAssoc.h` (282 lines) is now the base class, and
`HistoContainer.h:101` declares `class HistoContainer : public OneToManyAssoc<I, NHISTS*NBINS+1, SIZE>`.
`cudadev`'s `HistoContainer.h` also gained a `OneToManyAssocView` and separate
launch helpers (191 diff lines vs `cuda`).

This is a genuine restructure and it is **load-bearing** — `TrackSoAHeterogeneousT`,
the CA kernels, and the clusterizer all depend on it. Mojo has no inheritance, so
the base/derived split must be re-expressed (composition + trait, or parameterised
struct). Get this right before touching anything downstream.

- **New file:** `CUDACore/OneToManyAssoc.mojo`
- **Rewrite:** `CUDACore/HistoContainer.mojo`
- **Also new:** `CUDACore/FlexiStorage.mojo` (49 lines) — `OneToManyAssoc` uses it for
  the static/dynamic storage switch.

### C2. `GenericCachingAllocator` replaces notcub
`cudadev` dropped the two 600–750-line notcub allocators in favour of one
templated `GenericCachingAllocator.h` (336 lines) parameterised by traits:

```
getCachingDeviceAllocator.h:130  using CachingDeviceAllocator = GenericCachingAllocator<DeviceTraits>;
getCachingHostAllocator.h:98     using CachingHostAllocator   = GenericCachingAllocator<HostTraits>;
```

`mojocudatest` ported the *old* notcub pair
(`notcub/CachingDeviceAllocator.mojo`, `notcub/CachingHostAllocator.mojo`).
Those get replaced, not adapted.

- **New:** `CUDACore/GenericCachingAllocator.mojo`, `getCachingHostAllocator.mojo`
- **Rewrite:** `getCachingDeviceAllocator.mojo`
- **Delete:** the two `notcub/` files

### C3. Track SoA restructure
`cudadev` splits `PixelTrackHeterogeneous.h` (now 9 lines) into
`TrackSoAHeterogeneousT.h` (72) + `TrajectoryStateSoAT.h` (59).

Good news: `mojo-serial` **already** has this shape —
[`PixelTrackHeterogeneous.mojo`](../src/mojo-serial/MojoSerial/CUDADataFormats/PixelTrackHeterogeneous.mojo)
defines `TrackSoAT[S]` with a `HitContainer` alias and a separate
`TrajectoryStateSoA.mojo`. Two concrete deltas to fix:
- `hindex_type`: `mojo-serial` uses `DType.uint16`, `cudadev` uses `uint32_t`
- `HitContainer` must retarget the new `OneToManyAssoc` from C1

So this is C-bucket by dependency (blocked on C1), not by volume.

### C4. `HeterogeneousSoA` — re-add the device side
`cuda` and `cudadev` versions are **identical** (0 diff), but `serial` stripped 92
lines of device allocation. `mojo-serial`'s `HeterogeneousSoA.mojo` therefore has a
CPU-only hole where `cudaMalloc`/stream-ordered allocation belongs. Reinstate using
the `device_unique_ptr` machinery from `mojocudatest`.

Same pattern applies to `TrackingRecHit2DHeterogeneous` (`.cc` is new in `cudadev`).

### C5. CUDA→Mojo GPU primitives with no existing Mojo counterpart
None of these exist in either donor port:

| File | Lines | Note |
|------|-------|------|
| `CUDACore/launch.h` | 147 | kernel launch wrapper — maps onto `DeviceContext.enqueue_function`; central to every plugin |
| `CUDACore/radixSort.h` | 278 | device-side sort, used by vertex finder |
| `CUDACore/copyAsync.h` | 89 | H2D/D2H async copies |
| `CUDACore/memsetAsync.h` | 33 | |
| `CUDACore/ESProduct.h` | 109 | per-device event-setup product cache |
| `CUDACore/HostProduct.h` | 29 | |
| `CUDACore/HostAllocator.h` | 56 | pinned-host STL allocator |
| `CUDACore/host_noncached_unique_ptr.h` | 74 | |
| `CUDACore/requireDevices.{cc,h}` | 47 | |
| `Framework/CMSUnrollLoop.h` | 51 | `#pragma unroll` macros — Mojo has `@unroll`/`@parameter for` |
| `Framework/propagate_const_array.h` | 131 | const-propagating array wrapper |

`gpuAlgo1.mojo` notes that *"the Mojo CUDA layer in this project does not expose
cudaMemcpyAsync"* — so `copyAsync`/`memsetAsync` are a genuine gap that must be
built, not just translated. **Do `launch` + `copyAsync` + `memsetAsync` early**;
every plugin blocks on them.

### C7. `mojo-serial` silently de-atomicised the concurrent primitives — **re-atomicise, don't copy**

This is the trap that makes several "identical" files unsafe to copy, and it is not
visible from diffing the C++ or from counting Mojo dialect markers. `serial` removed
device concurrency; `mojo-serial` faithfully ported the removal. Copying those files
into a GPU port **compiles fine and silently races.**

Device-concurrency references (`atomicAdd`/`atomicCAS`/`atomicInc`/`__syncthreads`/
`__shfl`/`blockIdx`/`threadIdx`) in the `cudadev` header, versus what
`mojo-serial`'s port of the same thing contains:

| file | `cudadev` refs | `mojo-serial` |
|---|---|---|
| `CUDACore/prefixScan.h` | 25 | 0 |
| `CUDACore/SimpleVector.h` | 8 | 0 |
| `CUDACore/VecArray.h` | 4 | 0 |
| `CUDACore/AtomicPairCounter.h` | 1 | 0 |

`mojo-serial`'s `AtomicPairCounter.mojo` imports nothing but `bitcast` and `Typeable`
— the "atomic" in the name is all that survives. `SimpleVector.mojo` reaches for
`CUDACompat.atomicCAS`, whose body is a plain non-atomic read-modify-write:

```mojo
fn atomicAdd[T1: DType, //](a: UnsafePointer[Scalar[T1], mut=True], b: Scalar[T1]) -> Scalar[T1]:
    var ret = a[]
    a[] += b          # not atomic
    return ret
```

`mojo-serial`'s own `@deprecated` note on `struct CUDACompat` says as much: *"Any
methods using CUDACompat should be redirected to perform the regular operations since
we are not in a CUDA environment."* For this port the redirect target is the
opposite — **real device atomics from Mojo's `gpu` module.**

**Do not port `mojo-serial/CUDACore/CUDACompat.mojo`.** These four files are Bucket C,
scheduled with the Phase 2 container work, and each needs its atomics restored
against the `cudadev` header rather than transcribed from `mojo-serial`.

### C6. `CAHitNtupletGeneratorKernelsImpl.h` (167 diff lines, 607 total)
The CA algorithm core. Mix of C4-renames and real change. Needs line-by-line review
against `mojo-serial`'s `CAHitNtupletGeneratorKernels.mojo`. Same for
`GPUCACell.h` (184 diff lines) — this one has real structural change beyond renames.

---

## 5. Bucket D — Files that do not exist in Mojo at all

Grouped by why they're missing.

### D1. `.cu` kernel drivers — no CPU counterpart existed
`serial` had no separate kernel-launch translation unit, so nothing to port from.

```
plugin-SiPixelRecHits/PixelRecHitGPUKernel.{cu,h}          77 + 33 lines  (new in cudadev)
plugin-SiPixelClusterizer/SiPixelRawToClusterGPUKernel.cu
plugin-PixelTriplets/CAHitNtupletGeneratorKernels.cu
plugin-PixelTriplets/CAHitNtupletGeneratorKernelsAlloc.cu
plugin-PixelTriplets/BrokenLineFitOnGPU.cu
plugin-PixelTriplets/RiemannFitOnGPU.cu
plugin-PixelVertexFinding/gpuVertexFinder.cu
```

### D2. GPU-side data formats
`serial` used plain `*SoA` host structs; `cudadev` uses device-resident `*CUDA` ones.

```
CUDADataFormats/SiPixelDigisCUDA.{cc,h}          48 + 85
CUDADataFormats/SiPixelClustersCUDA.{cc,h}       19 + 63
CUDADataFormats/SiPixelDigiErrorsCUDA.{cc,h}     40 + 42
CUDADataFormats/BeamSpotCUDA.h                        33
CUDADataFormats/TrackingRecHit2DHeterogeneous.cc      44
CUDADataFormats/TrackSoAHeterogeneousT.h              72   (see C3)
CUDADataFormats/TrajectoryStateSoAT.h                 59   (see C3)
```

### D3. Cabling map rework — renamed *and* reworked
`mojo-serial` has `SiPixelFedCablingMapGPU{,Wrapper}.mojo`. `cudadev` renamed and
reworked these into:

```
CondFormats/SiPixelROCsStatusAndMapping.h                    25
CondFormats/SiPixelROCsStatusAndMappingWrapper.{cc,h}        57 + 46
plugin-SiPixelClusterizer/SiPixelROCsStatusAndMappingWrapperESProducer.cc   43
```
Not a pure rename — check field-by-field before reusing the Mojo version.

### D4. Small new `DataFormats` headers
```
DataFormats/DetId.h                  93
DataFormats/SiPixelDigiConstants.h   60
DataFormats/SiPixelErrorCompact.h    13
DataFormats/SiPixelFormatterErrors.h 11
DataFormats/PixelSubdetector.h       14
DataFormats/SiStripEnums.h           10
```
Trivial — ~200 lines total, mostly enums and constants. Good warm-up tasks.

### D5. Plugins with no `mojo-serial` counterpart
`mojo-serial` has 6 plugin dirs; `cudadev` has 8. Missing entirely:
```
plugin-PixelTrackFitting/PixelTrackSoAFromCUDA.cc     69   (D2H track transfer)
plugin-SiPixelRawToDigi/SiPixelDigisSoAFromCUDA.cc    72   (D2H digi transfer)
plugin-PixelVertexFinding/PixelVertexSoAFromCUDA.cc   49   (D2H vertex transfer)
plugin-BeamSpotProducer/BeamSpotToCUDA.cc             44   (H2D beamspot transfer)
plugin-SiPixelClusterizer/SiPixelClusterThresholds.h  14
```
The four `*FromCUDA`/`ToCUDA` producers are the host↔device boundary — they are
short but they are where `ScopedContextAcquire`/`ScopedContextProduce` actually get
exercised. All four block on C5 (`copyAsync`).

### D6. Not ported, deliberately deferred
```
Framework/Worker.{cc,h}, WaitingTaskList.{cc,h}, FunctorTask.h,
RunningAverage.h, hardware_pause.h
bin/PluginManager.{cc,h}, SharedLibrary.{cc,h}
```
Neither donor port has these. `mojo-serial` and `mojocudatest` both sidestep dynamic
plugin loading (`SharedLibrary`/`PluginManager`) by static registration — keep doing
that. Revisit only if TBB-style concurrent scheduling becomes a goal.

`cudadev/test/` (26 files) is out of scope for the initial port.

---

## 6. Cross-cutting decision: import convention

The two donor ports use **incompatible** module conventions:

- `mojo-serial`: package-qualified — `from MojoSerial.CUDACore.HistoContainer import ...`,
  plugin dirs use underscores (`plugin_PixelTriplets`)
- `mojocudatest`: flat, driven by `-I` flags in `pixi.toml` — `from CUDACompat import ...`,
  `-I . -I CUDACORE/cms/CUDA -I CUDACORE/cms/notcub -I plugin-Test1`, plugin dirs use hyphens

**Decision: adopt the `mojo-serial` *layout* convention** (package-qualified,
underscored plugin dirs) **while taking `mojocudatest`'s Mojo *dialect*** (§0). The
two choices are independent: layout is about import paths, dialect is about language
version. Cost: rewrite imports in the ~25 CUDACore files coming from `mojocudatest`.
Do it in one pass up front, not file-by-file.

Rewrite rules for `mojocudatest/CUDACORE/cms/CUDA/*` → `MojoCudaDev/CUDACore/*`:
```
from <Sibling> import          →  from MojoCudaDev.CUDACore.<Sibling> import
from Framework.X import        →  from MojoCudaDev.Framework.X import
from MojoBridge.X import       →  from MojoCudaDev.MojoBridge.X import
from std.… import              →  unchanged (stdlib)
```

### `MojoBridge` must be merged, not chosen

Neither donor's `MojoBridge` is a superset:

| | `mojo-serial` | `mojocudatest` |
|---|---|---|
| `Matrix`, `Vector`, `SymmetricEigen`, `Print`, `Static`, `Timer` | ✅ needed by every fit/algorithm | ❌ absent |
| `ConcurrentQueue` | ❌ absent | ✅ needed by `StreamCache`/`EventCache` |
| `DTypes`: CUDA aliases (`cudaError_t`, `cudaSuccess`, `cudaErrorMemoryAllocation`, `cudaErrorNotReady`, `cudaEventDisableTiming`, `CUresult`, `CUDA_SUCCESS`, `cudaErrorName()`, `cudaErrorMessage()`) | ❌ absent | ✅ required by `ScopedContext`, `cudaCheck` |
| `DTypes`: `TypeableInt`, `TypeableUInt` | — | ✅ |
| `OrderedMap` / `OrderedMultiSet` | diverged (58 / 34 lines) | diverged |

`DTypes.mojo` alone differs by 246 lines between the two. Take the **union**: for the
four modules present in both, the §0 rule gives `mojocudatest`'s version outright;
the six `mojo-serial`-only maths/utility modules are additive. If any symbol
`mojo-serial` relies on turns out to be missing from `mojocudatest`'s `DTypes`, add
it rather than reverting the file.

Also note `mojocudatest` carries a **duplicate** `MojoBridge` at
`CUDACORE/cms/CUDA/MojoBridge/` (a 20-line-stale copy of the top-level one, missing
`TypeableInt`/`TypeableUInt`). Collapse it — there must be exactly one
`MojoCudaDev/MojoBridge/`.

Proposed layout:
```
src/mojo-cudadev/
├── pixi.toml            # mojo 25.5.*, platforms = ["linux-64"]
├── Makefile.deps
└── MojoCudaDev/
    ├── CUDACore/  CUDADataFormats/  CondFormats/  DataFormats/
    ├── Framework/ Geometry/  MojoBridge/  bin/
    └── plugin_BeamSpotProducer/ plugin_PixelTrackFitting/ plugin_PixelTriplets/
        plugin_PixelVertexFinding/ plugin_SiPixelClusterizer/
        plugin_SiPixelRawToDigi/ plugin_SiPixelRecHits/ plugin_Validation/
```

Note: the top-level [`Makefile`](../Makefile) has **no** Mojo targets — the existing
Mojo ports build through `pixi` only. Follow that; don't try to wire into the C++
build system.

---

## 7. Ordering

Dependency-driven, not size-driven.

**Phase 0 — toolchain spike, then skeleton**

*0a. Spike (blocking, needs a linux-64 + NVIDIA host — see §0).* Before moving any
code: pick the Mojo version, then take the **five smallest** `mojo-serial` Framework
files and compile them under it. That measures the real cost of the dialect
migration — specifically whether `ImplicitlyCopyable` must be added to all ~38
`(Copyable, …)` structs and whether the 598 `UnsafePointer` uses need explicit
origins. Everything after this is estimated off that number. Do not skip it.

*0b. Skeleton.* Create `src/mojo-cudadev/`, `pixi.toml`, package dirs. Merge
`MojoBridge` (§6). Move Bucket A1 + A2 in and normalise imports. Get an empty
`main.mojo` compiling and running.

**Phase 1 — GPU primitives (blocks everything)**
C5: `launch`, `copyAsync`, `memsetAsync`, `HostAllocator`, `host_noncached_unique_ptr`,
`ESProduct`, `HostProduct`. Then C2 (`GenericCachingAllocator`).
Validate by re-running the `mojocudatest` `gpuAlgo1` path on the new allocator.

**Phase 2 — container restructure (blocks all algorithms)**
C1: `OneToManyAssoc` + `FlexiStorage` + `HistoContainer` rewrite. Then C3 (Track SoA),
C4 (`HeterogeneousSoA` device side). Port D4 (trivial DataFormats headers) in parallel —
they're independent and make good low-risk warm-ups.

**Phase 3 — data formats and conditions**
D2 (`*CUDA` data formats), D3 (`SiPixelROCsStatusAndMapping`).

**Phase 4 — algorithms, one plugin at a time, each end-to-end**
Order chosen so each stage's output validates before the next consumes it:
1. `SiPixelClusterizer` (+ D1 `SiPixelRawToClusterGPUKernel.cu`, B-bucket renames)
2. `SiPixelRecHits` (+ D1 `PixelRecHitGPUKernel`)
3. `PixelTriplets` — the big one: B-bucket rename sweep on BrokenLine/RiemannFit/FitUtils/
   CAConstants, then C6 (`CAHitNtupletGeneratorKernelsImpl`, `GPUCACell`), then D1 kernels
4. `PixelVertexFinding` (mostly bucket A — barely changed from `cuda`)
5. `BeamSpotProducer`

**Phase 5 — host↔device boundary and validation**
D5 (`*FromCUDA`/`ToCUDA` producers), then `plugin-Validation` (`CountValidator`,
`HistoValidator`) to check numerical agreement against `mojo-serial` and against
C++ `cudadev` output.

---

## 8. Risk register

| Risk | Why it matters | Mitigation |
|------|----------------|------------|
| **Mojo dialect split between donors (§0)** | The two ports don't compile together; `mojo-serial` is 22k lines of the older dialect and is the source of every algorithm | Phase 0a spike. This is the top risk — it gates the schedule and the estimate |
| **De-atomicised primitives (§C7)** | `mojo-serial` stripped device atomics; copied files compile and race silently. Affects `prefixScan`, `SimpleVector`, `VecArray`, `AtomicPairCounter` and every kernel | Before copying any `mojo-serial` file, grep its `cudadev` counterpart for `atomic*`/`__syncthreads`/`__shfl`/`blockIdx`/`threadIdx`. Non-zero ⇒ Bucket C, not a copy |
| **No linux-64 / NVIDIA host available** | Nothing in this plan is compile-verified on the current machine (Darwin arm64, no `mojo`, no GPU) | Secure a build host before Phase 1; until then treat all output as unverified |
| `copyAsync`/`memsetAsync` have no Mojo backing | `gpuAlgo1.mojo` explicitly worked around the absence of `cudaMemcpyAsync`; the four D5 transfer producers cannot be written without it | Spike this in Phase 1 before committing to the schedule. If `DeviceContext` genuinely can't express it, the whole host↔device boundary design changes |
| `OneToManyAssoc` base/derived split has no Mojo equivalent | Mojo has no class inheritance; C1 blocks Phases 2–5 | Prototype the composition/trait shape first, in isolation, against `mojo-serial`'s existing tests |
| "Rename-only" assumption on BrokenLine/RiemannFit | 595 and 557 diff lines — if any of it is real numerics, silent wrong results | Normalise renames then re-diff; validate fit output against `mojo-serial` on the same events |
| `hindex_type` uint16 vs uint32 mismatch | Silent overflow at high occupancy | Fix in C3, add an explicit assertion |
| `cudadev` is a moving upstream target | Rebase pain mid-port | Pin the current `cudadev` commit; treat drift as a separate follow-up |

---

## 9. Rough effort shape

| Bucket | Files | Character |
|--------|-------|-----------|
| A — reuse as-is | ~90–100 | import rewrite only |
| B — minimal change | ~15 | mechanical rename sweep, verify don't trust |
| C — real work | ~20 | design required; C1, C2, C5 are the hard core |
| D — doesn't exist | ~30 | new Mojo, but mostly small; D1 kernels are the substance |

The port is front-loaded: **C1, C2 and C5 are the schedule.** Once `launch`,
`copyAsync`, `GenericCachingAllocator` and `OneToManyAssoc` are solid, the remaining
~150 files are largely transcription against two working reference implementations.

---

## 10. Progress

### Done — Phase 0b, `mojocudatest`-sourced half (56 files)

`src/mojo-cudadev/` created with `pixi.toml` and the §6 package layout. Everything
below was copied from `mojocudatest` and had its flat `-I` imports rewritten to
package-qualified `MojoCudaDev.*` form. All of it is already in the target (newer)
Mojo dialect, so none of it is blocked on the Phase 0a spike.

| Destination | Files | Source |
|---|---|---|
| `MojoCudaDev/CUDACore/` | 25 | `CUDACORE/cms/CUDA/` + `allocator/` + `notcub/`, flattened |
| `MojoCudaDev/Framework/` | 15 | `Framework/` |
| `MojoCudaDev/DataFormats/` | 8 | `DataFormats/` |
| `MojoCudaDev/MojoBridge/` | 5 | `MojoBridge/` (top-level; the stale `CUDACORE/cms/CUDA/MojoBridge/` duplicate was dropped) |
| `MojoCudaDev/bin/` | 5 | `Bin/` (renamed lowercase to match layout) + root `main.mojo` |

Verified: zero residual non-stdlib flat imports across the tree.
Reproducible via `scratchpad/phase0b_import.sh`.

**Unverified:** not compiled. No `mojo` toolchain and no NVIDIA GPU on this machine
(§0). The import rewrite is textual and mechanically checked, nothing more.

### Done — trivial `mojo-serial` copies (10 files)

Copied with `from MojoSerial.` → `from MojoCudaDev.`, chosen because their `cudadev`
counterparts have **zero** device-concurrency references (§C7) and the files carry no
old-dialect constructs beyond trait conformances:

```
MojoBridge/   Print.mojo  SymmetricEigen.mojo  Vector.mojo  Matrix.mojo
DataFormats/  ApproxAtan2.mojo  SiPixelRawDataError.mojo  BeamSpotPOD.mojo
              PixelErrors.mojo  SOARotation.mojo
Geometry/     Phase1PixelTopology.mojo
```

`MojoBridge/DTypes.mojo` gained `hex_to_float` (+ a `from memory import bitcast`),
which `SOARotation` needs and `mojocudatest`'s `DTypes` lacked — added rather than
reverting the file, per §6. All imports in the copied files now resolve against
modules present in the tree.

**Reverted during this step:** `CUDACore/{SimpleVector, VecArray, AtomicPairCounter,
PrefixScan}.mojo` were copied, then backed out on discovering §C7. They are Bucket C.

### Done — re-atomicised primitives (§C7), 3 of 4

Method: take the `mojo-serial` file as the base, add the atomicity back against the
`cudadev` header. Working notes and the full verification checklist live in
[MojoCudaDevAtomics.md](MojoCudaDevAtomics.md).

| file | atomic sites | note |
|---|---|---|
| `CUDACore/CUDAAtomics.mojo` | — | new; all unverified API surface confined here |
| `CUDACore/SimpleVector.mojo` | 6 | `push_back`, `extend`, `shrink` |
| `CUDACore/VecArray.mojo` | 2 | `push_back` |
| `CUDACore/AtomicPairCounter.mojo` | 1 | `add`, on the 64-bit view |

Diffed against `mojo-serial`: nothing changed but the atomicity and the import lines.

**`CUDACore/PrefixScan.mojo` is not started.** It needs `__shfl_up_sync`,
`__ballot_sync`, `__syncthreads`, `__threadfence`, dynamic shared memory and
`extern __shared__` — none of which exist in either donor port. It belongs with the
Phase 1 GPU primitives, not with this pass.

**Bug found in `mojo-serial` and fixed here:** `SimpleVector.extend()` undid `1` on
the failure path where `cudadev/CUDACore/SimpleVector.h:89` undoes the full `size`.
Live bug upstream, independent of atomics.

### Done — `DataFormats` closed out (§D4)

Six headers written fresh in the target dialect (no `mojo-serial` counterpart
existed), all with zero device-concurrency refs:

```
DetId.mojo (+ Detector)   SiPixelDigiConstants.mojo (+ getLink/getROC/getADC/
PixelSubdetector.mojo       getCol/getRow/getDCol/getPxId)
SiStripEnums.mojo         SiPixelErrorCompact.mojo   SiPixelFormatterErrors.mojo
```

Plus `SiPixelDigisSoA.mojo` copied from `mojo-serial` — its 14-line
`serial`→`cudadev` diff turned out to be **comments only**, so it is a true Bucket A
copy; the comments were carried over.

**Superseded:** `DataFormats/PixelErrors.mojo` was copied earlier and has been
removed. `cudadev` split `serial`'s `PixelErrors.h` into `SiPixelErrorCompact.h` +
`SiPixelFormatterErrors.h`, renaming `PixelErrorCompact` → `SiPixelErrorCompact` and
`PixelFormatterErrors` → `SiPixelFormatterErrors`. The `Dict[UInt, ...]` key type is
kept from `mojo-serial` rather than narrowing to `UInt32` as the C++ does, since that
choice is presumably load-bearing for Mojo's `KeyElement` requirement.

`DataFormats/` is now complete against `cudadev`: `fed_header.h` and `fed_trailer.h`
have no separate Mojo file because `mojo-serial` folded their fields directly into
`FEDHeader.mojo` / `FEDTrailer.mojo`, and `approx_atan2.h` is `ApproxAtan2.mojo`.

Still held back for the dialect spike: `MojoBridge/Static.mojo` (`.alloc(`) and
`MojoBridge/Timer.mojo` (old `__moveinit__` signature).

### Not done — blocked on the Phase 0a spike

- `MojoBridge/Matrix.mojo`, `Vector.mojo`, `SymmetricEigen.mojo` and `Print.mojo` were
  copied in an earlier pass (see "Done — trivial `mojo-serial` copies" above) and are
  confirmed present in the tree. Only `Timer.mojo` (124 lines, old `__moveinit__`
  signature) and `Static.mojo` (63 lines, contains one of the 11 `.alloc(` sites)
  remain unported.
- Everything else sourced from `mojo-serial` — i.e. every algorithm.

### Done — Phase 0a spike, for real (2026-07-29)

This machine turned out to be linux-64 (not the Darwin arm64 assumed in §0), and
`pixi` + a real Mojo toolchain are available. The spike in §7 finally ran.

**Build unblocked.** `mojo-cudadev/pixi.toml` pinned `mojo = "25.5.*"`, copied
verbatim from `mojo-serial`'s manifest — but `mojo-serial` only resolves because it
ships a `pixi.lock` frozen from when `25.5.0` was still resolvable. With no lock file,
`mojo-cudadev` re-solves against today's channel index and `25.5.*` no longer exists
there at all (confirms §0's "renumbered upstream" note, and this is not a platform
issue — it reproduces identically on linux-64). Fixed by pinning to
`max = "26.2.*"` (channels trimmed to `max` + `conda-forge`), which resolves to
**Mojo 0.26.2.0**, the exact version `mojocudatest` itself locks to — i.e. this now
actually implements the §0 decision ("standardise on `mojocudatest`'s dialect")
instead of accidentally floating to whatever's newest (a fresh, unpinned solve grabs
**1.0.0b2**, which has removed `fn` entirely in favour of `def` and restructured
stdlib import paths — a far bigger jump than anything in §0's dialect table. Do not
let `mojo-cudadev` float; keep it pinned to `26.2.*` until a deliberate version bump).

**No physical GPU is needed to compile-verify.** `mojo build` normally auto-detects
the local GPU architecture and fails hard (`constraint failed: Unknown GPU
architecture detected`) on a GPU-less machine. Passing `--target-accelerator sm_80`
(or any arch from `mojo build --print-supported-accelerators`) bypasses
auto-detection entirely. Under this flag, `mojo build -I . MojoCudaDev/bin/main.mojo`
compiles and links the **entire current tree** (all of Bucket A1/A2, the mojo-serial
copies, and the re-atomicised C7 primitives) into a real ELF binary with **zero
compile errors** — only 27 warnings, almost all `alias`→`comptime` deprecation
notices. This means §0's core fear (major structural breakage from the dialect gap —
`ImplicitlyCopyable`, explicit origins on the 598 `UnsafePointer` uses) did **not**
materialise for anything currently in the tree. Re-run this build after every future
batch of files lands; it costs nothing and this was previously assumed impossible on
any machine without an NVIDIA card.

**Two real runtime bugs found by actually running the binary** (not visible from
reading — this is exactly why §7 said the spike "cannot be resolved by reading, needs
a compiler"):

1. **`bin/StreamSchedule.mojo` `run()` called `exit(0)` directly** the instant
   `maxEvents` was reached, hard-killing the whole process mid-`runToCompletion()` —
   skipping `endJob()`, the wall/CPU timing, and the final throughput print in
   `main.mojo`. The C++ original (`cudadev/bin/StreamSchedule.cc`) has no such call;
   `Source::produce()`/`Source.mojo`'s `produce()` already signals "done" by
   returning a null pointer, and the `while ptr != null` loop was already written to
   stop on its own. Fixed: replaced `exit(0)` with `return`. Every `--maxEvents` run
   (the project's primary benchmarking mode) was silently producing zero output
   before this fix.

2. **`fn __del__(var self):` is silently miscompiled into infinite self-recursion in
   Mojo 0.26.2.0.** Reproduced in isolation with a two-line struct (`var d: Int`,
   `__del__` that only prints) — destroying it via `_ = x^` *or* plain end-of-scope
   both crash with a stack overflow, same recursive return address repeating
   hundreds of times. The compiler explicitly rejects `fn __del__(mut self):` with
   `'self' argument must be passed as 'deinit'`, but does not apply the same check to
   `var self` — it accepts it and generates broken code instead of raising the same
   error. `owned self` no longer parses at all (dead keyword: "expected ')' in
   argument list"). The fix is `fn __del__(deinit self):`, and it is universal — this
   is the same `var`→`deinit` shift §0 already documented for `__moveinit__`'s second
   parameter (`var other: Self` → `deinit take: Self`), just unrecorded for `__del__`.
   Fixed tree-wide: every `fn __del__(var self):` in `src/mojo-cudadev` (17 sites,
   14 files: `Event`, `ESPluginFactory`, `SharedEventPtr`, `StreamSchedule`,
   `ReusableObjectHolder` ×2, `ProductBase`, `EventCache`, `ScopedContext` ×3,
   `CachingDeviceAllocator`, `EventSetup`, `StreamCache`, `ScopedSetDevice`,
   `host_unique_ptr`, `getCachingDeviceAllocator`) → `deinit self`. Verified: a full
   `--maxEvents N` run now completes end-to-end and prints the final throughput line
   (previously the process spun at 100% CPU across 16 threads forever after
   `endJob()`, having recursed into `ESPluginFactory.Registry.__del__` without limit).

**New risk found, not yet fixed — latent breakage hidden by non-reachability.**
`Framework/WaitingTaskHolder.mojo` and `WaitingTaskWithArenaHolder.mojo` (the
mojocudatest-only "async pieces" from §2/A2) compile as part of the full
`main.mojo` build with zero errors, but only because **nothing in the current
producer-less execution path actually calls into their method bodies** — Mojo
appears not to fully type-check an unreached method body. Compiling
`WaitingTaskHolder.mojo` directly surfaces real errors: `fn __del__(owned self):`
(dead keyword, same as above), `fn __copyinit__(out self, other: Self):` (must be
named `copy`), `fn __moveinit__(out self, var other: Self):` (must be `deinit take`),
plus an unrelated direct-field-access error (`m_group[]` needs `self.` prefix) and a
`self`-shape check `wm_group`. These are exactly the "written but unverified" files
in Bucket A2 the doc already flagged as `mojocudatest`-only, and the same dialect
migration cost §0 estimated for `mojo-serial` applies to them too — the difference is
they weren't caught because they're not yet on the path any registered plugin
exercises. **Do not trust "it's in Bucket A, mojocudatest already has it" for these
two files without compiling them directly** — and re-check every other currently
"working" file the same way once a real plugin starts exercising more of the call
graph, since the same masking could be hiding elsewhere.

### Done — Phase 1, `CUDACore/launch.mojo` (2026-07-29)

Ported `LaunchParameters` (grid/block dims, shared memory, stream) matching the C++
struct field-for-field. **`cms::cuda::launch(kernel, config, args...)` itself has no
Mojo equivalent as a callable wrapper, and this is a hard language constraint, not a
missing-API gap** — confirmed empirically with three escalating tests before writing
the file:

1. A generic `launch[func_type, //, func: func_type, *args](...)` wrapping
   `stream.enqueue_function[func, func](...)` fails because `DeviceStream` (unlike
   `DeviceContext`) doesn't even have the `[func, signature_func]`-style overload —
   only the pre-compiled-`DeviceFunction` one.
2. Switching to `ctx.compile_function[func, func]()` (which does have that overload)
   and forwarding `func` through the wrapper still fails:
   `'compile_function' parameter 'signature_func' has 'def(*args: *) -> None' type,
   but value has type 'func_type'`. The compiler needs to introspect the kernel's
   *own* concrete parameter-type list at the call site to derive `declared_arg_types`;
   a kernel forwarded through an abstract generic parameter has already had that
   concrete signature erased, so it can never satisfy this no matter how the wrapper
   is shaped.
3. The direct (non-wrapped) call `ctx.compile_function[kernel, kernel]()` then
   `stream.enqueue_function(compiled, args..., grid_dim=, block_dim=,
   shared_mem_bytes=)` compiles clean and, at runtime on this GPU-less box, raises a
   clean caught exception (`Failed to open library "libnvidia-ml.so.1"`) instead of
   crashing — confirming the pattern is sound, just not wrappable.

Pushed on this further to see if *any* wrapper shape works, since the config-struct
result felt like giving up early — tried accepting an already-compiled
`DeviceFunction` instead of a bare kernel, which sidesteps #2's problem entirely
(the wrapper then only has to forward the *argument* pack, not the kernel). Got
further — `DeviceFunction[_, _, target=_, compile_options=_,
_ptxas_info_verbose=_]` (Mojo's wildcard/unbind syntax) accepts any concrete
`DeviceFunction` generically without needing to spell out its internal type
parameters — but hit a second, fully independent wall: forwarding the wrapper's own
`*args: *actual_arg_types` pack into `stream.enqueue_function(compiled, *args, ...)`
errors **"unpacked arguments are not supported yet."** That is an explicit
not-implemented from the compiler, not a type mismatch to design around. So there
are two independent blockers, not one — a generic `launch()` is unreachable by any
design in this Mojo version, confirmed both ways.

This costs nothing in practice: every `cudadev` call site already names its kernel
literally, never through indirection, so the C++ template's only real job was
argument-type deduction, which Mojo gets for free from `compile_function` at the
literal call site anyway. The call-site idiom (documented in the file's header
comment) is two lines instead of one function call — `compile_function` on the
`DeviceContext`, `enqueue_function` on the target `DeviceStream`/`LaunchParameters.stream`.

`launch_cooperative` (`cudaLaunchCooperativeKernel`) is **not ported** —
`grep`-confirmed `cudadev` only defines it, never calls it, and both blockers above
apply to it identically (also confirmed empirically). If it's ever needed, cooperative
launch is just an ordinary `enqueue_function` call carrying an extra
`LaunchAttribute(LaunchAttributeID.COOPERATIVE, LaunchAttributeValue(True))` —
verified to compile and fail gracefully the same way as the base pattern.

Verified: `launch.mojo` compiles standalone, compiles as part of the full
`main.mojo` tree (still 0 errors), and a full test exercising
`LaunchParameters` → `compile_function` → `enqueue_function` end-to-end compiles
and fails gracefully (not a crash) at runtime without a GPU.

### Done — Phase 1, `CUDACore/copyAsync.mojo` and `memsetAsync.mojo` (2026-07-29)

Same investigative rigor as `launch.mojo`, and it paid off the same way: neither
`cudaMemcpyAsync` nor `cudaMemsetAsync` can be expressed via any high-level Mojo API
that targets an arbitrary, caller-chosen stream. Checked every copy-capable surface —
`DeviceContext.enqueue_copy` (9 overloads), `DeviceContext.enqueue_memset` (2),
`DeviceBuffer`/`HostBuffer.enqueue_copy_to`/`enqueue_copy_from`/`enqueue_fill` — every
one rejects a `stream=` keyword outright (`"unknown keyword argument: 'stream'"`), and
`std.gpu.host` has exactly 9 modules total, all checked, no lower-level path hiding
elsewhere. This matches `gpuAlgo1.mojo`'s own comment ("the Mojo CUDA layer... does
not expose cudaMemcpyAsync") — a previously-known gap, not a new discovery, but now
fully mapped instead of just noted.

**Resolution: kernel-based copy/fill**, launched via the same `compile_function` +
`stream.enqueue_function` idiom as `launch.mojo` — that path *does* support an
explicit stream. Verified the one real open question first: can a single kernel
signature accept both a device pointer and a host-buffer pointer without an
address-space mismatch? Yes — `HostBuffer.unsafe_ptr()` and `DeviceBuffer.unsafe_ptr()`
both return the identical plain `UnsafePointer[Scalar[dtype], MutAnyOrigin]` (Mojo
doesn't distinguish host/device address spaces at the type level here), and a copy
kernel taking that type compiles and launches cleanly with pointers from either side.

Element-wise copy kernel needs `T: ImplicitlyCopyable & Movable & ImplicitlyDestructible`
for `dst[i] = src[i]` to compile (the same three-trait bundle §0's dialect table
already flagged, just discovered here empirically via the compiler's own error
messages rather than by inspection). The memset kernel only bitcasts to `UInt8` and
never touches `T`'s own copy/move semantics, so it needs no bound beyond `AnyType`.

Ported the four `copyAsync` overloads that only need `device_unique_ptr`/
`host_unique_ptr` (both single- and multi-element, both directions) and the one
`memsetAsync` overload `cudadev` actually calls (`SiPixelDigiErrorsCUDA.cc:14` —
the array/`int value`/`nelements` form). **Not ported**, matching the
`launch_cooperative` precedent of not writing unverifiable code with no real call
site: the `host::noncached::unique_ptr` and `std::vector<T, HostAllocator<T>>`
`copyAsync` overloads (types don't exist in this tree yet), the `propagate_const_array`
overload (same), and the single-element `T value` `memsetAsync` overload (relies on
C++ implicitly narrowing a `T` to `int`; `cudadev` never calls it).

**Found and fixed a real, previously-undetected bug via the same "unexercised code"
mechanism as `WaitingTaskHolder`** (§ above): `CUDACore/allocate_host.mojo` — both
`allocate_host_raw` (line 51) and `free_host_raw` (line 61) computed a pointer's
integer identity as `UInt(ptr.address)`, which doesn't type-check
(`.address` produces a raw MLIR pointer value, not something `UInt` accepts). This
file was already compiling as part of the full `main.mojo` tree with 0 errors —
because nothing in the current producer-less path ever calls `make_host_unique` or
lets a `host_unique_ptr` go out of scope, so `free_host_raw`'s body was never actually
type-checked. `copyAsync.mojo` is the first file in this port to use
`host_unique_ptr` in a way that forces its destructor chain
(`OwnedPointer.__del__` → `_HostAllocation.__del__` → `free_host_raw`) to compile,
and it immediately surfaced the bug. Fixed by matching the already-correct,
already-exercised sibling pattern in `allocate_device.mojo`: `UInt(Int(ptr))` instead
of `UInt(ptr.address)`, at both sites. **Same lesson as before: "0 errors in the full
build" only covers what's actually reachable from `main()` today — expect more of
these as each new file starts exercising previously-dormant corners of the tree.**

Verified: both files compile standalone and as part of the full tree (still 0
errors), and a full integration test — `make_device_unique`/`make_host_unique` →
all four `copyAsync` overloads → `memsetAsync` → real `CUDAStreamType`/`currentDevice()`
plumbing — compiles and runs to completion, raising a clean caught
`cudaError_t(600)` (`cudaErrorNotReady`, this project's own error code, not a raw
crash) on this GPU-less machine, exiting 0.

**Not done yet**: `HostAllocator.mojo`, `host_noncached_unique_ptr.mojo`,
`ESProduct.mojo`, `HostProduct.mojo`, `requireDevices.mojo`, then C2
(`GenericCachingAllocator.mojo`, `getCachingHostAllocator.mojo`, and rewriting the
still-stale notcub-based `getCachingDeviceAllocator.mojo`). Once `HostAllocator.mojo`
and `host_noncached_unique_ptr.mojo` land, come back and add `copyAsync`'s four
remaining overloads.

`HostAllocator.h` itself is likely **not going to be ported as a literal file**: its
only real usage in `cudadev` is as a `std::vector<T, HostAllocator<T>>` custom
allocator (one field, `SiPixelROCsStatusAndMappingWrapper.h:28`); Mojo's `List[T]` has
no custom-allocator concept, and the underlying capability (sized pinned-host
allocation) is already covered by `host_unique_ptr.make_host_unique[T](n, ...)`.
Revisit when that CondFormats file is actually ported (Phase 3) rather than writing an
unused allocator-interface shell now.

### Milestone — a real NVIDIA GPU became available mid-port (2026-07-30)

Everything above this line was written and verified on a machine with no GPU at all
(§0's "Darwin arm64, no `mojo`, no NVIDIA GPU" — this later turned out to actually be
a different, GPU-having Linux machine the whole time, just without a working driver
yet; see below). Every "compiles and fails gracefully without a GPU" claim above is
now superseded by an actual run against real hardware.

**The hardware was there the whole time.** `lspci` showed an NVIDIA GeForce RTX 4060
(AD107, Ada Lovelace, `sm_89`) at `01:00.0`. What was missing was the driver:
`nouveau` (no CUDA support) was bound to the card, and the proprietary driver was
stuck mid-install — 18 packages in dpkg's `iU` (unpacked, not configured) state, the
DKMS source present at `/usr/src/nvidia-current-550.163.01` but the `dkms` tool itself
not installed, so the kernel module never got built.

Getting a fully working driver took two rounds:
1. Debian's own repo (`trixie/non-free`) installed and configured driver `550.163.01`
   cleanly (`nvidia-smi` worked, all kernel modules loaded) — but Mojo's MAX runtime
   (this toolchain, `max`/`mojo-compiler` 0.26.2.0) refused to construct a
   `DeviceContext` against it: `"MAX doesn't support your current NVIDIA GPU driver.
   MAX requires a minimum driver version of 580 and CUDA version 13.0. Your driver
   version is 550.163.01"`. Confirmed this wasn't a code bug in `CUDACompat.mojo` by
   testing bare `DeviceContext()` and `DeviceContext(api="cuda")` directly — both fail
   identically, so it's purely a driver-floor check, unrelated to the `api=` argument.
2. **Before reaching for a driver upgrade, checked whether the problem was actually
   solvable for free**: Mojo 25.5.0 (`mojo-serial`'s already-installed pin) constructs
   a `DeviceContext` and runs a real kernel successfully against driver `550.163.01` —
   confirmed with an actual `+1` kernel over 1000 floats, correct results read back.
   So the *driver* wasn't strictly the blocker — the *newer Mojo toolchain's* minimum-
   driver check was. Given the choice between migrating the whole port to the older
   dialect (reopening §0's dialect-migration cost) versus upgrading the driver, decided
   to upgrade the driver and keep 0.26.2.0 (the user's call, not a default) —
   nothing already written needed to change either way once the driver was current.
3. Added NVIDIA's own repo for Debian 13 (`cuda-keyring` + the `debian13/x86_64`
   suite) and installed `cuda-drivers`, landing on `610.43.02` (CUDA 13.3 — comfortably
   past the 580/13.0 floor). Two real snags along the way, worth recording since they
   ate the most time: `wget` doesn't exist on this system (`curl -O` works fine, no
   other change needed), and the very first attempt at this second round silently
   didn't run at all — always verify system state directly (`dpkg -l`, `apt-cache
   madison`, `/var/log/apt/history.log`) rather than trusting "I ran it" at face value
   when a command depends on multiple prior steps having actually executed.

**Result — real, on-hardware verification, not just compilation, for everything
Phase 0–1 had only compile-checked before:**
- A bare `compile_function` + `enqueue_function` kernel launch (the exact
  `launch.mojo` idiom) now runs for real: a `+1` kernel over 1000 floats on the actual
  RTX 4060, correct results read back (`host_view[0] = 1.0`, `host_view[999] =
  1000.0`).
- `copyAsync.mojo` + `memsetAsync.mojo`, exercised through the project's own
  `make_device_unique`/`make_host_unique`/`CUDAStreamType` plumbing (not a bare
  standalone test) — real host→device→host round trip with real data, and a real
  device-side zero-fill, both verified correct by reading values back
  (`h_ptr[50] = 50.0` after a zero/copy/round-trip; `h_ptr2[7] = 0.0` after
  `memsetAsync`).
- The full `main.mojo` binary (`--empty`, 0 producers) still runs cleanly end to end
  against the new driver.

Going forward, every Phase 1+ file should be verified the same way this milestone
was proven — real execution with a real result check, not just "0 compile errors" —
now that the machine can actually do it. `--target-accelerator sm_89` matches this
specific card (Ada Lovelace); drop it entirely (or use `mojo run`) now that a real
default device is present, rather than cross-compiling blind.

### Done — Phase 1, `CUDACore/host_noncached_unique_ptr.mojo` (2026-07-30)

Same shape of problem as `copyAsync`/`memsetAsync`: the C++ side takes a CUDA
host-alloc flags argument (all 3 real call sites — `BeamSpotToCUDA.cc`,
`SiPixelRawToClusterGPUKernel.cu` ×2 — pass `cudaHostAllocWriteCombined`), and Mojo's
high-level `DeviceContext.enqueue_create_host_buffer` has no flags parameter at all.
Unlike `launch`/`copyAsync`, this time the flag was worth actually getting right
rather than documenting as dropped, since it directly affects the very call sites
that need it. With real hardware now available, tested whether dropping to raw FFI
could recover it, rather than assuming Mojo's high-level API is the only option:

- `libcudart.so` (the CUDA *runtime* API library, e.g. `cudaHostAlloc` itself) is not
  installed on this system at all.
- `libcuda.so.1` (the CUDA *driver* API library, e.g. `cuMemHostAlloc`) **is**
  present, and its flag values are numerically identical to the runtime API's
  (`cudaHostAllocWriteCombined == CU_MEMHOSTALLOC_WRITECOMBINED == 0x04` — guaranteed
  by NVIDIA's own API design, not a coincidence), so the existing `DTypes.mojo`
  constant could be reused directly with no new aliases needed.
- Mojo's `std.ffi.external_call["cuMemHostAlloc", Int32](...)` compiles and, with a
  `DeviceContext(api="cuda")` already constructed first (a real CUDA context must be
  current — confirmed by the exact failure when skipped: `CUresult 3`,
  `CUDA_ERROR_NOT_INITIALIZED`, matching every real call site's actual usage inside an
  already-active `CUDAAppContext`), the call **succeeds for real**: alloc with the
  write-combined flag, write, read back, free — all `CUDA_SUCCESS`.

This is the first file in this port to link against a CUDA library directly instead
of going through `std.gpu.host`. It needed one build-system change: `pixi.toml`'s
`run`/`build`/`verify` tasks now pass `-Xlinker -l:libcuda.so.1` — the *exact*
versioned filename, not `-lcuda`, because only `libcuda.so.1` exists on this system
(no unversioned `libcuda.so` dev-symlink), and the linker's `-l` flag only searches
for the unversioned name unless given the exact filename via `-l:`.

Also hit, and fixed: bare `T` inside the struct's own method bodies needs `Self.T` in
this dialect (matches §0's dialect table) — caught because standalone compilation
(`mojo build ... host_noncached_unique_ptr.mojo --emit object`) checks the file for
real, unlike the full-tree `verify`, which reported 0 errors throughout even before
the fix. **Confirms, again: a clean full-tree build does not mean an unreferenced new
file is actually being checked** — the same lesson as `WaitingTaskHolder` and
`allocate_host.mojo`, now with a concrete mitigation: compile every new file
standalone before trusting it, don't rely on the full-tree result until something
actually imports it.

`HostAllocator.mojo` itself remains not-ported (see the note above this milestone) —
unaffected by this file, since the two serve different real call sites.

### Done — Phase 1 (C5), `CUDACore/HostProduct.mojo` (2026-07-30)

Ported as `Optional[HostUniquePtr[T]]` / `Optional[OwnedPointer[T]]` (both wrapped in
`Optional` since neither is nullable on its own, unlike C++'s `unique_ptr`), `get()`
preferring the pinned slot. `get` takes `mut self`, not `self` — `OwnedPointer.unsafe_ptr()`'s
returned origin tracks the receiver's mutability, and an immutable receiver can't
produce the `MutAnyOrigin` this codebase's other `get()` methods return.
`__moveinit__` needs `deinit take: Self`, not `var take: Self` — the latter doesn't
support moving out two separate fields of the same parameter in this dialect.
Verified correct on real hardware for both storage paths individually.

**Found a real, reproducible bug in `host_unique_ptr.mojo`'s destructor** while
testing this: freeing a `host_unique_ptr` hangs forever acquiring the allocator's
lock, but only when no other `DeviceContext` is alive anywhere else in the process
(never the case in the real port — every plugin runs inside an already-active
`CUDAAppContext`). Calling `free_host_raw` directly on the named `_AllocateHostState`
variable never hangs; only going through the pointer captured by
`make_host_unique`/stored in `_HostAllocation` does. Root cause not fully understood.
Noted at the site in `host_unique_ptr.mojo`. Also fixed, separately, while
investigating: both `allocate_host.mojo` and `allocate_device.mojo` stored their
buffer without keeping its creating `DeviceContext` alive alongside it — a real bug
in its own right, just not the one causing this hang.

A narrower, still-unresolved variant: two *different* `HostProduct[T]`
instantiations (e.g. `HostProduct[Float32]` and `HostProduct[Int32]`) alive at the
same time in one program hang on exit even with another `DeviceContext` kept alive,
though each alone doesn't. No real call site needs two different `HostProduct[T]`s
simultaneously, so this is flagged rather than chased further for now.

### Done — C2, `CachingDeviceAllocator.mojo` + `CachingHostAllocator.mojo` fixed and verified (2026-07-30)

Went looking for the `GenericCachingAllocator<Traits>` unification C2 calls for, and found
both concrete allocators **already existed** (837 and ~640 lines) — a prior pass had
already ported the old non-generic notcub-style pair by hand. First confirmed Mojo can
even express the generic-over-traits pattern before considering the full merge (a
trait with an associated type + a struct generic over that trait — spiked in isolation,
works cleanly, two different concrete conformances driving the same generic struct).
But then found `DeviceAllocate` doesn't compile at all (`out` parameter combined with
an explicit return type — invalid in this dialect) — meaning neither file had ever
actually been exercised. Decided to fix what's broken and keep the two allocators
separate rather than risk a large merge on top of code that had never compiled once
(user's call, not a default).

Both files needed far more than the one signature: `.address` used directly in
comparisons/`String()` (needs `Int(ptr)`); a function with two `out` parameters
(invalid — needs a `Tuple` return instead); bare `alias`/`comptime` struct members
accessed without `Self.` from within the struct's own methods; `mut foo: Int`
parameters called with a `comptime` constant directly (needs an intermediate `var`
binding — a `mut` parameter needs an assignable local, not a constant); a
non-parametrized-trait requirement (`trait Foo[T: Bound]:` is an explicit compiler
TODO in this dialect — the working substitute is a trait with an **associated type**
member instead, `alias ItemType: Bound`, which the earlier spike had already proven
works). `CachingHostAllocator.mojo` additionally had: nested structs (`BlockDescriptor`
and both comparators declared inside the allocator struct — not supported at all in
this dialect, all four had to become top-level, matching how `CachingDeviceAllocator.mojo`
already did it correctly); `@fieldwise_init` on a struct with a non-movable
`BlockingSpinLock` field (needs an explicit `__moveinit__` instead, again matching the
sibling file); a `Sized` conformance with no `__len__` (dropped, unused); a `let`
keyword (not a Mojo keyword at all -- `var`); `\Host` string literals (invalid escape
sequence -- missing the `t` in `\tHost`); and a `lower_bound()` call missing its
required search-key argument.

`MojoBridge/OrderedMultiSet.mojo` itself needed the trait-parametrization fix too —
its `Compare` type parameter had no way to be constrained to match the stored item
type `T` (Mojo can't express that without parametrized traits either), so `T` was
dropped as a separate parameter entirely and the stored item type is now derived
directly from `Compare.ItemType` — one type in play instead of two that could
mismatch.

Verified beyond compiling: a real allocate → free → allocate-again cycle on actual
hardware, for both allocators, confirming the *same address* is returned the second
time (real bin-cache reuse working, not just "didn't crash").

Also added `CachingHostAllocator.__del__` (calling `FreeAllCached()`), which didn't
exist before — the `skip_cleanup` field was already there with a comment describing
exactly this destructor behavior, just no destructor to use it; `CachingDeviceAllocator`
already had the matching one. Slightly beyond a pure bug-fix, flagged as such rather
than silently bundled in.

Not done: the actual `GenericCachingAllocator<Traits>` unification remains undone —
`CachingDeviceAllocator.mojo` and `CachingHostAllocator.mojo` stay separate, duplicated
files, per the decision above. `getCachingDeviceAllocator.mojo`'s factory (the
still-stale notcub-era one, `CUDATEST_DISABLE_CACHING_ALLOCATOR` etc.) is now backed by
a working `CachingDeviceAllocator`; an equivalent `getCachingHostAllocator.mojo` factory
does not exist yet and would be the natural next step if `CachingHostAllocator` needs
a shared/singleton instance the way the device one does.

### Done — `getCachingHostAllocator.mojo` (2026-08-03)

Wrote the singleton-factory wrapper around `CachingHostAllocator`, mirroring
`getCachingDeviceAllocator.mojo`'s `_CachingAllocatorState` pattern. Smaller than the
device version since C++'s `getCachingHostAllocator.h` has no `Policy`/`_selectPolicy`
of its own (host allocation has no sync/async/caching switch) and reuses
`binGrowth`/`minBin`/`maxBin`/`debug`/`minCachedBytes`/`_formatBinSize` from the device
file rather than redefining them — matching C++'s own `#include` reuse. Kept the
spinlock in `_CachingHostAllocatorState`: confirmed (by reading `allocate_device.cc`/
`allocate_host.cc`) that `getCachingDeviceAllocator()`/`getCachingHostAllocator()` are
called from `cms::cuda::allocate_device`/`allocate_host`, the core per-allocation
entry points hit from every EDM worker thread concurrently — so the lock is
replicating a real race C++'s function-local `static` guards for free, not
defensive boilerplate. Verified standalone-compiling.

Also separately ported the `memsetAsync` single-element overload (`memsetAsync.h:11-18`,
previously skipped for having no real call site) using Mojo's `Intable` trait as the
equivalent of C++'s implicit `T→int` narrowing into `cudaMemsetAsync`. Verified on real
hardware: filling a 4-byte `Int32` with value `7` read back as `117901063 = 0x07070707`,
confirming the byte-fill (not element-assign) semantics match C++.

Explored (not adopted) a `Kernel`-trait alternative design for the still-unported
generic `launch()` wrapper: a trait requiring a `run_kernel` method, with each distinct
kernel call getting its own conforming struct holding its args as fields. Spiked and
confirmed **this does work** in this dialect (real GPU run, `WriteValueCall` struct
writing 99 through a generic `launch[K: Kernel](...)` entry point) — it sidesteps both
walls documented in `launch.mojo` because the kernel is named literally inside the
concrete struct's own method, and args are listed explicitly rather than forwarded as a
pack. Left as a documented idea rather than adopted: it trades the current
zero-extra-types inline idiom (already used at every real call site) for one
`Kernel`-conforming struct per distinct kernel, for a more C++-shaped uniform call site
— not clearly worth it unless something wants to hold a kernel launch as a first-class
value.

### Done — Phase 1, `CUDACore/ESProduct.mojo` (2026-08-03)

Last unported Phase 1 item besides `HostAllocator`. Per-GPU-device lazy-fill cache for
EventSetup/conditions data (`ESProduct.h`): a private `Item` struct (mutex, `SharedEventPtr`,
filling-stream marker, atomic `filled` flag, the `T` data) held one per device, plus the
double-checked-locking `dataForCurrentDeviceAsync` that either returns already-filled data
lock-free, waits on the recorded CUDA event if someone else is filling, or does the fill
itself and records the event.

Nearly all the plumbing already existed in this port: `SharedEventPtr`, `EventCache`,
`ScopedSetDevice`, `eventWorkHasCompleted`, `cudaStreamWaitEvent`/`cudaEventRecord`,
`deviceCount`/`currentDevice`, and `Atomic` (already used in `SharedEventPtr.mojo`). `Item`
is heap-allocated per device (`alloc[Item[T]](1)`, stored as `List[UnsafePointer[...]]`)
rather than held inline in a `List`, mirroring `EventCache._buildSlots` exactly — needed
because `BlockingSpinLock` can't live inline in a `List` (non-movable/non-copyable, the
same constraint hit earlier in `CachingHostAllocator`).

The `transferAsync` callback (a generic higher-order function parameter in C++) was the one
open design question, since generic higher-order functions have broken twice before in this
port (`launch()`). Spiked in isolation first: a plain `fn(mut T, StreamType) raises -> None`
parameter works fine here, because unlike `launch()` this isn't forwarding a *variadic pack*
into a nested call — it's a single function value, called directly. No wall hit.

`Item.filling_stream`'s "no one is filling" sentinel is `cudaStreamDefault` (`CUDAStreamType()`),
deliberately preserving the same ambiguity C++'s `nullptr` has (CUDA's default-stream handle
*is* the null pointer value) — a latent landmine already present in the C++ original (if the
first filler uses the default stream, a second thread on a different real stream would also
see "not filling yet" and could re-trigger the transfer), left as-is rather than silently
changed into different behavior.

Two real, previously-hidden bugs found and fixed along the way, both in `ScopedSetDevice.mojo`
— never called from anywhere in this port before, so never actually compiled:
- `__source_location()` isn't a real declaration in this dialect. The correct form is
  `source_location()` from `std.reflection`, returning a `SourceLocation` with plain
  **fields** `.file_name`/`.line` (not methods) — every other call site in this codebase
  already used the (correct-but-cruder) literal-string-plus-`0` convention instead, which is
  how the mismatch went unnoticed.
- `self.prevDevice_` was passed as a `mut` out-param to `cudaGetDevice` before ever being
  assigned — Mojo's definite-initialization checker doesn't treat that as initializing the
  field, unlike a direct `self.field = value` assignment. Needed an explicit `self.prevDevice_
  = -1` first, matching the pattern the same function already used correctly for its local
  `currentDevice` variable a few lines down.

Also hit, fixed the same way as in the caching allocators: `Atomic` has no plain
`store(self, value)` instance method, only a low-level `store(ptr, value)` free-function form
— used `fetch_add` instead (safe here since the value is confirmed `0` under the lock right
before).

Verified: standalone compile, plus a real-hardware instantiation (`ESProduct[Payload]`, a
`transferAsync` closure setting a field to `777`, two calls returning `777` both times).
Coverage caveat: the test only exercises two of the four logical branches (first-filler, and
already-filled fast path) — the "someone else is filling on a different stream" and
"someone else was filling but their event has since completed" branches are real code paths
but need genuine concurrent access from another thread to exercise, which this single-threaded
smoke test doesn't attempt.

Not done: `HostAllocator.mojo` — the last remaining Phase 1 item. No natural Mojo equivalent
of a pluggable `std::allocator` for `List[T]`, needs its own shape decision before writing.

### Done — Phase 1, `CUDACore/HostAllocator.mojo` (2026-08-03) — Phase 1 fully closed out

C++'s `HostAllocator<T, FLAGS>` is a pure `std::allocator`-concept class (`allocate`/
`deallocate`/`rebind`) meant to be plugged into `std::vector<T, HostAllocator<T>>`. Mojo's
`List[T]` has no pluggable-allocator mechanism, so before writing anything, checked the one
real call site (`SiPixelROCsStatusAndMappingWrapper.cc:20,24,45-48`) to see what's actually
needed: not a dynamically-growing vector at all — `modToUnpDefault(modToUnp.size())` is
constructed once with a known size, filled by a single elementwise copy, then read via
`.data()`/`.size()` for a `cudaMemcpyAsync`. No `push_back`, no resizing, ever.

Asked the user how to shape/name this given the mismatch between "C++ allocator" and "what's
actually needed" (a small fixed-size buffer) — decided to keep the file named
`HostAllocator.mojo` for 1:1 traceability to `HostAllocator.h`, even though the Mojo type is
really a thin fixed-size buffer, not an allocator. Implemented as a wrapper around the already-
existing `make_host_noncached_unique[T]` (reusing its genuine pinned-allocation-with-flags
rather than re-deriving the raw `cuMemHostAlloc`/`cuMemFreeHost` FFI again), adding `.size()`/
`.data()`/`__getitem__`/`__setitem__` on top. `bad_alloc` (`HostAllocator.h:12-20`) wasn't
ported either — no custom exception type exists anywhere else in this port, every allocation
failure already raises a generic `Error` with a descriptive message.

Needed `T: AnyType & Copyable & ImplicitlyCopyable` (not just `Copyable`) for `__getitem__`/
`__setitem__` to type-check — matches the `_CopyElement` bound already established in
`copyAsync.mojo`.

Verified on real hardware, but hit one linking wrinkle worth recording: `mojo run` (JIT) failed
to resolve `cuMemHostAlloc`/`cuMemFreeHost` even with `-Xlinker -l:libcuda.so.1` in the same
place that has worked all session for `mojo build`-produced binaries — JIT mode doesn't resolve
FFI `external_call` symbols the same way a real linked binary does. Building to a binary and
running that directly worked. Separately, the built binary's first run hit a genuine runtime
error, `cuMemHostAlloc failed with CUresult 3` (`CUDA_ERROR_NOT_INITIALIZED`) — the raw driver
API needs some `std.gpu.host.DeviceContext` to have initialized CUDA first in the same process;
this port's raw-FFI files have always been exercised so far alongside other DeviceContext-using
code, so this dependency had never surfaced as its own test requirement before. Fixed by
constructing a `DeviceContext(api="cuda")` before using `HostAllocator` in the test — a test-only
fix, not a change to `HostAllocator.mojo` itself. Final run: `size: 5`, `sum of elements: 100`
(0+10+20+30+40, confirming per-element set/get through the pinned buffer), non-null `data()`.

This closes out Phase 1 ("GPU primitives — blocks everything") entirely.

### Found and fixed — real file corruption in `CachingDeviceAllocator.mojo` (2026-08-04)

While building a `gpuAlgo1`-style validation test (see below), `mojo build` failed with a wall
of "unexpected character"/"unterminated string" errors inside `CachingDeviceAllocator.mojo`,
at lines nothing in this session had touched recently. Reading the file (both via the editor
tool and via `sed`) showed several paragraphs of prose sitting inside a broken string literal —
text that reads almost verbatim like a chat summary of the C2 allocator work reported earlier
in this session ("Both CachingDeviceAllocator.mojo (837 lines) and CachingHostAllocator.mojo
(~640 lines) went from 'never once compiled' to fully working...").

First reaction was to suspect tool-output tampering / prompt injection, since `git diff HEAD`
and `md5sum` both showed the working-tree file was byte-identical to what was already
committed — which felt like it ruled out real corruption. That reasoning was wrong: "identical
to HEAD" only proves the working tree matches the last commit, not that the commit itself is
clean. Checking against the true base (`git show 69bdf03a:...`, the commit before any of this
session's changes) settled it — the base version has correct code at that exact location, and
`grep -a -b` against both the raw file and the git blob confirmed the same corrupted bytes at
the same offset in both, with matching file sizes. So this was a real, self-inflicted bug: at
some point during the earlier (summarized) C2 work, a tool call ended up writing chat-summary
prose into the file in place of the tail of one debug-print string literal, and it was
committed without being caught (the standalone compile check done at the time evidently ran
before this happened, and nothing re-touched this exact file afterward to catch it).

Scanned the rest of `src/mojo-cudadev` for the same pattern (`grep` for a few distinctive
phrases, plus a blanket search for any `.mojo` line over 300 characters) — isolated to this one
spot in this one file. Fixed by restoring the string literal to match the identical pattern used
elsewhere in the same file (`"available blocks cached (" + String(totals.free) + ...`). Verified
clean: standalone compile, zero remaining matches for the leaked text.

Lesson for future verification: "working tree matches HEAD" is not evidence of correctness by
itself — always check against a known-good point (base commit, or an actual compile/run) before
concluding a file is fine, and before escalating to "this might be external tampering."

### Done — Phase 1 realistic validation: `gpuAlgo1` workload through `CachingDeviceAllocator` (2026-08-04)

Closes the one item from Phase 1's own definition that hadn't been done yet: "Validate by
re-running the `mojocudatest` `gpuAlgo1` path on the new allocator." `gpuAlgo1.mojo`
(`src/mojocudatest/plugin-Test1/gpuAlgo1.mojo`) is test/demo-only content with no `cudadev`
counterpart, so this wasn't ported into the permanent tree — it's a scratchpad validation
exercise that re-runs the same workload shape (multiple differently-sized buffers: two 1000-float
vectors, a 1000-float result vector, a 1,000,000-float matrix; init/vectorAdd/vectorProd/
matrixMulVector kernels) through `CachingDeviceAllocator.DeviceAllocate`/`DeviceFree` instead of
the plain `device_unique_ptr`/`_AllocateDeviceState` path `gpuAlgo1` originally used, run twice
back to back.

Result on real hardware: both passes produced numerically identical output
(`997501700000000.0`, `1.9930078e+17`), and all four buffers landed at the exact same address on
the second pass — genuine bin-cache reuse holding up under a realistic multi-size, multi-buffer,
multi-kernel pattern, not just the single-buffer alloc/free/alloc-again smoke test done earlier.
This is the strongest evidence yet that C2 is solid.

## Phase 2 (C1) — `OneToManyAssoc` / `HistoContainer` inversion, started (2026-08-04)

Scoped before writing anything. Read `OneToManyAssoc.h` (282 lines), `HistoContainer.h` (174
lines), `FlexiStorage.h` (49 lines), and the existing `mojo-serial/.../HistoContainer.mojo` (361
lines, pre-inversion shape where `OneToManyAssoc` is just an alias for `HistoContainer`) as
reference. Two problems have no direct Mojo equivalent:

1. **No inheritance** — `HistoContainer : public OneToManyAssoc<I, NHISTS*NBINS+1, SIZE>` in
   C++. Decided: composition (`HistoContainer` holds a `OneToManyAssoc` field), with the base's
   ~15 public methods forwarded explicitly on `HistoContainer` to preserve C++'s call-site
   ergonomics (`histo.finalize()`, not `histo.base.finalize()`), since downstream consumers
   (`TrackSoAHeterogeneousT`, CA kernels, clusterizer) use both freely today.

2. **`FlexiStorage<I,S>` switches between a fixed inline array and runtime pointer+capacity
   storage via C++ template specialization on `S == -1`.** Checked first whether
   `mojo-serial`'s approach (skip `FlexiStorage` entirely, inline a plain `InlineArray` field
   into `HistoContainer` directly) would work here — it doesn't, because `OneToManyAssoc` is
   used **standalone**, not only through `HistoContainer`: `TrackSoAHeterogeneousT.h:19`
   (`HitContainer`) and `CAConstants.h:76-78` (`TuplesContainer`, `HitToTuple` — which uses
   `ONES = -1`, genuinely runtime-sized) both instantiate it directly. So a real, independently
   usable `FlexiStorage` is required.

   Spiked whether Mojo can express the C++ specialization as a single `FlexiStorage[I, S]` type
   (matching C++'s 2-parameter shape) rather than two separate types unified by a trait: declare
   *both* possible field sets (`InlineArray[I, max(S,1)]` and `UnsafePointer[I]` + capacity)
   unconditionally, and have every method pick which one to use via `@parameter if S >= 0`.
   Confirmed on real instantiation, both modes: fixed (`S=5`) gave capacity `5`, sum `100`;
   dynamic (`S=-1`, backed by a `List`'s buffer via `.init(ptr, 8)`) gave capacity `8`, sum
   `2800`. This works and keeps every downstream alias site at the same 3-parameter shape as
   C++ (`OneToManyAssoc<I, ONES, SIZE>`), at the cost of a small always-present dummy field in
   whichever mode isn't active (a 1-element placeholder array in dynamic mode, an unused
   pointer+int in fixed mode).

### Done — `CUDACore/FlexiStorage.mojo`

Written from the validated spike shape. `capacity()`, `__getitem__`, `__setitem__`, `data()`
all `@parameter if Self.S >= 0`-branch between the two storage modes; `init(ptr, capacity)`
exists unconditionally (a no-op to call in fixed mode, since `_fixed` is already valid storage
without it — C++ only defines `init` on the `S == -1` specialization at all). Verified:
standalone compile, plus a real run against the actual file (not just the spike) — same
results as the spike, fixed capacity `5`/sum `100`, dynamic capacity `8`/sum `2800`, non-null
`data()`.

Not done yet: `OneToManyAssoc.mojo` (needs real atomics for `atomicIncrement`/`atomicDecrement`/
`add`, per §C7's "re-atomicise, don't copy" — `mojo-serial` did these as plain non-atomic
`+=`/`-=`) and `HistoContainer.mojo` (composition wrapper forwarding the base's methods).

### Done — `CUDACore/PrefixScan.mojo`, blocking `OneToManyAssoc.finalize()` (2026-08-04)

Full write-up in [MojoCudaDevAtomics.md](MojoCudaDevAtomics.md) (where this was already tracked
as part of §C7's re-atomicising work). Short version: verified `CUDAAtomics.mojo`'s atomics for
real on hardware (found and fixed a real `mut=True` syntax bug there too, never-before-compiled),
then built `blockPrefixScan` on `std.gpu.primitives.warp`/`gpu.sync`/shared memory — none used
anywhere in this port before. Hit and root-caused a genuine Mojo/MAX deadlock along the way:
divergent warp-shuffle (a subset of a warp calling `vote()`/masked `shuffle_up` while the rest
have branched away, standard and well-defined in raw CUDA) hangs in this dialect; fixed by having
every lane always participate in the shuffle uniformly, gating only the read/write of results.
Verified correct across 12 boundary cases, both device and host code paths. This was the
long pole for `OneToManyAssoc.finalize()` — C1 can now continue with `OneToManyAssoc.mojo` itself.

`PrefixScan.mojo` was refined twice more after this, both from direct feedback rather than bugs
found independently: `ws` was put back as a caller-supplied parameter (matching C++ exactly,
including re-adding `assert(ws)` — it can be null again now that it's not internally allocated),
and the block-uniform round-count loop was replaced with a per-*warp*-uniform condition
(`warpBase = first - laneId`, identical across all 32 lanes of a warp) — closer to C++'s actual
`while i < size` shape, and strictly better than the round-count version since entirely-irrelevant
warps now skip their loop altogether instead of doing one wasted padding round. Both re-verified
across the full boundary matrix (including a many-empty-warps case, `size=17`/`block_size=256`)
plus the caller-supplied-`ws` shape — still 15/15, no hangs.

### Done — `CUDACore/OneToManyAssoc.mojo` (2026-08-05)

Ported with real atomics (`CUDAAtomics.mojo`) and `blockPrefixScan` (`PrefixScan.mojo`), now that
both are verified. Composition, not inheritance, per the earlier C1 scoping. `OneToManyAssocView`
is ported too (just a plain generic struct, no `launch()`-style problem) — see below for its final
shape, which changed from the original associated-type-trait design once that turned out to crash
the compiler. `index_type` (not the raw `I` parameter) is used everywhere C++ itself uses
`index_type` rather than `I` directly (`content` field, `fill`, `begin`/`end`), matching the C++
source's own naming choice rather than my first draft's shortcut.

**`OneToManyAssoc.initStorage` — initially deferred, then resolved (2026-08-07).** Two
*independent* compiler crashes were found here, not one, and both are now understood and fixed:

- **Bug A**: `FlexiStorage`'s `capacity()`/`__getitem__`/`__setitem__`/`data()` used
  `@parameter if Self.S >= 0: ... else: ...` to pick fixed-array vs. pointer+runtime-size storage.
  Merely having an `@parameter if` present in the *type of a field* of a self-referential struct
  crashed the compiler at the time — even in a method that's never called, regardless of which
  branch the comptime condition would take. Originally fixed by replacing all four with plain
  runtime `if`. **Update (2026-08-07): re-tested against the final `View[Assoc]` design below and
  it no longer crashes** — `@parameter if` is restored in `FlexiStorage.mojo`, confirmed via a
  standalone compile of `OneToManyAssoc.mojo`, a full project-wide cross-compile, and a hardware
  re-run, all clean. So Bug A's trigger wasn't "any `@parameter if` in a field's type" in
  isolation after all — it depended on some interaction with the *old* `View` shape (the
  trait/associated-type projection, or the two-parameter `[Assoc, I]` intermediate) that no longer
  exists. Left unresolved exactly which part of the old shape was necessary; not worth
  re-bisecting now that the design that needed the workaround is gone.
- **Bug B**: separately, *using* a value of a type projected through a trait's associated type on
  a self-referential parameter (`Self.Assoc.Counter`/`Self.Assoc.index_type` where `Assoc = Self`)
  — as opposed to merely storing/passing it — also crashes the compiler. This is what actually
  blocked `initStorage`'s body (`self.content.init(view.contentStorage, ...)`), independent of
  Bug A and of alias declaration order (both explored first; neither was the real cause). Isolated
  to a two-line minimal repro: field access/discard is fine, but type-checking the value against
  anything (a call argument, `==`, `print()`) triggers it.

Fixed without dropping the associated-type shape entirely — the crash is specifically about
*projecting through* `Self.Assoc.X` (using a value of that type), not about the trait bound
merely existing. Final design: `OneToManyAssocLike` is back (`Counter`/`index_type` associated
types, matching C++'s `using Counter = typename Assoc::Counter` textually), and
`OneToManyAssocView[Assoc: OneToManyAssocLike]` is bound to it — but no field is ever *typed* via
`Self.Assoc.Counter`/`.index_type`. `offStorage` is hardcoded `UnsafePointer[UInt32, ...]`
(`Counter` never varies). `contentStorage` is `UnsafePointer[NoneType, ...]` (untyped/opaque),
`.bitcast[Self.index_type]()` right where it's actually used, inside `OneToManyAssoc.initStorage`
itself — where `index_type` is a plain leaf comptime value on `Self`, not a projection through
`Assoc`. `OneToManyAssoc.View` is `OneToManyAssocView[Self]`. Verified empirically, not assumed,
across three spikes: reviving the *original* Bug-B design (`Self.Assoc.index_type` actually typing
a field, then used inside `initStorage`) still segfaults identically even after Bug A's fix
(`selfref_view_from_assoc.mojo`) — so Bug B is independent of Bug A, not a side effect of it.
Binding the trait but never projecting through it compiles clean
(`selfref_view_trait_unused.mojo`). Two earlier intermediate shapes were tried and abandoned along
the way: threading `I` through as a second explicit parameter alongside `Assoc` (crash-free, but
redundant once the opaque-pointer approach was found), and dropping the trait bound entirely in
favor of `Assoc: Movable` (also crash-free, but further from the C++ shape than the trait-bound,
projection-free version above). Confirmed in isolation before touching the real file each time
(`selfref_view_no_I_param.mojo`, `selfref_view_from_assoc.mojo`, `selfref_view_trait_unused.mojo`),
then applied and re-verified via standalone compile.

A third intermediate shape was tried and reverted, worth recording since it produced a real
process lesson, not just a design dead end: making `initStorage` *independently* generic over its
own `Assoc` parameter (never writing `Self`/`Self.View` anywhere in `initStorage`'s own
signature or body — `Assoc` inferred from the argument at each call site instead), with
`contentStorage` properly typed as `Assoc.index_type` rather than opaque `NoneType`. This compiled
clean standalone (`selfref_view_indep_bitcast.mojo`, both with an explicit `[Assoc]` type argument
and with it inferred) — genuinely no crash, because the elaborator never has to resolve `Self`
inside `Self`'s own body at all. But wired into the real file and exercised from an actual call
site (`onetomany_content_dyn_test.mojo`, which calls `initStorage` for real), it crashed with the
identical Bug-B signature. **Standalone compilation of a file containing only a generic method's
*definition* does not fully monomorphize it — a real call site is required to trigger this class of
crash.** The three-spike trail above happened to dodge this because each spike's `main()` also
*called* the method — this fourth one didn't get caught until wiring into the test suite, which is
why the hardware tests (not just standalone/project compiles) are load-bearing verification here,
not a formality. Also tried, before reverting: removing `FlexiStorage`'s `@parameter if` (Bug A's
fix) again, on the chance it was an additional contributing factor for this specific crash — it
isn't; the call-site crash persists identically either way, confirming this is purely Bug B, with
no interaction with Bug A. Reverted back to the trait-bound + opaque-pointer + `Self.View` design
above (both `OneToManyAssoc.mojo` and `FlexiStorage.mojo`), re-confirmed via all three hardware
call sites (`onetomany_off_fill_test.mojo`, `onetomany_content_dyn_test.mojo`,
`onetomany_fixed_test.mojo`) plus a full project cross-compile — this remains the final, working
design.

**`setContentStorage()` closes the resulting type-safety gap on the write side.** Since
`contentStorage` is opaque `UnsafePointer[NoneType, ...]`, a caller doing
`view.contentStorage = anything.bitcast[NoneType]()` had zero static type checking — any type
could be bitcast in with nothing catching a mismatch before it silently produced UB inside
`initStorage`'s own `.bitcast[Self.index_type]()` on the read side. Fixed by adding
`fn setContentStorage(mut self, ptr: UnsafePointer[Self.Assoc.index_type, MutAnyOrigin])` as a
method *on `OneToManyAssocView` itself* (not on the self-referential `OneToManyAssoc`), doing the
bitcast internally. This is safe from Bug B: `View` isn't the self-referential struct (only
`OneToManyAssoc` is), and this method is never invoked from inside `OneToManyAssoc`'s own
elaboration, only from ordinary caller code once `Assoc` is already a concrete type. Confirmed via
a real call site, not just a bare definition (`selfref_view_typed_setter.mojo`), then applied and
re-verified: standalone compile, the dynamic-content hardware test (now using
`view.setContentStorage(content_storage)` instead of the raw bitcast, same correct output), and a
negative test confirming a mismatched pointer type (`Float32` against an `Int32`-keyed `Assoc`) is
rejected with a precise compile error, not a crash or silent UB.

**TODO, not yet addressed:** `setContentStorage()` is a convention, not an enforced boundary --
confirmed by directly testing `view.contentStorage = wrong.bitcast[NoneType]()`, which compiles
with zero errors, bypassing the setter entirely. Mojo has no field-privacy mechanism in this
dialect (`private var x: Int32` doesn't even parse -- tested directly), so `contentStorage` and
every other field on `OneToManyAssocView` are always public and directly writable from any caller.
Not a regression versus C++ (its `View` is an equally public plain aggregate, no setter there
either), but worth a cheap mitigation later: renaming to `_contentStorage` to match
`FlexiStorage`'s own `_fixed`/`_ptr`/`_capacity` underscore convention, signaling "internal, go
through `setContentStorage()`" even though it can't be truly enforced. Deferred, not blocking.

`initStorage` itself is now ported (`OneToManyAssoc.mojo:97-115`), matching the C++ body 1:1
(guarded by plain `if Self.ctCapacity() < 0` / `if Self.ctNOnes() < 0` — these were written plain
from the start, not converted; left as-is since `@parameter if` was only restored in
`FlexiStorage.mojo`). Verified on real hardware with `ONES=-1, SIZE=10` (dynamic off-storage, exercising
`off.init()`): `initStorage` → `zero()` → `count()` → `finalize()` → `fill()`, each step's result
read back from the external device buffer and checked against the expected counting-sort values —
all exact. (First verification attempt read results back by copying the whole `Assoc` struct to
host and calling `.size()`/`.size(b)` on it — that segfaulted, but it was a test-harness bug, not
a port bug: `off._ptr` in dynamic mode holds a raw device address, and dereferencing a device
pointer from host code segfaults in Mojo exactly as it would in C++. Fixed by reading the external
off-storage buffer directly instead.)

Other real, previously-unexercised bugs found and fixed along the way (this file's own imports
pulled in code that had never been compiled in this exact form before):
- `AtomicPairCounter.mojo`: `@register_passable("trivial")` is removed in this dialect version —
  needs `TrivialRegisterPassable` in the conformance list instead (matches the pattern already
  used by `DTypes.mojo`'s `TypeableInt`/`TypeableUInt` etc.).
- `InlineArray[UInt32, 5](0x2, 0xC, ...)` (the direct-variadic-args constructor, used successfully
  in `mojo-serial`'s own `HistoContainer.ilog2`) doesn't parse in this exact toolchain — that
  constructor overload requires an internal `__list_literal__` marker meant to back Mojo's `[...]`
  list-literal syntax specifically, not direct invocation. Fixed via
  `alias b: InlineArray[UInt32, 5] = [0x2, 0xC, 0xF0, 0xFF00, 0xFFFF0000]` instead.
- In the verification test (not the port itself): comparing two `UnsafePointer`s directly with
  `!=` triggered a parser failure that cascaded into bogus "excess indentation" errors several
  lines later — the same "one real problem, garbled errors downstream" pattern as the injected-text
  incident earlier, just a different root cause this time. Fixed by comparing `Int(ptr)` values
  instead, matching the address-comparison idiom already used elsewhere in this port.

Verified end-to-end on real hardware with a full counting-sort cycle: 10 items distributed into 3
bins (2/3/5), `zero()` → `count()` (atomic increments) → `finalize[block_size=64](ws)` (real
`blockPrefixScan`, converting counts to offsets) → `fill()` (atomic decrements placing items) →
read back via `size()`/`size(b)`/`begin(b)`/`end(b)`. Every value matched exactly: `size()=10`,
`size(0..2)=2,3,5`, and iterating `begin(b)..end(b)` for bins 0 and 2 produced exactly 2 and 5
elements. This is the first time in this port that a real GPU-computed prefix scan has been used
to drive an actual data structure's reuse/placement logic, not just tested in isolation. Re-run
unchanged after both `View` redesigns above — still 4/4.

`initStorage`'s dynamic-storage path is separately verified in both directions: `ONES=-1, SIZE=10`
(dynamic `off`, exercising `off.init()`) with the same counting-sort cycle read back from the
external off-buffer directly (`off = [0, 2, 5, 10]` after `finalize`+`fill`, exact); and
`ONES=4, SIZE=-1` (dynamic `content`, exercising `content.init()` and the `.bitcast[index_type]()`
call specifically) with the external content-buffer read back and each item found in the correct
bin's slice (bin 0 → indices 0-1, bin 1 → 2-4, bin 2 → 5-9, order within a bin unconstrained since
`fill` places via atomic decrement). Both re-run and re-confirmed after the final single-parameter
`View[Assoc]` redesign, plus a full project-wide cross-compile of `main.mojo` to catch collateral
breakage — none found.

### Done — `CUDACore/cudastdAlgorithm.mojo` and `CUDACore/HistoContainer.mojo` (2026-08-07)

`cudastdAlgorithm.mojo`: only `upper_bound` ported (the only one `HistoContainer`'s kernels
actually call; no custom `Compare` parameter, matching what's used). One real bug caught before
it shipped: `Int(ptr)` subtraction is byte-scaled, not element-scaled (confirmed directly —
`Int(p+5) - Int(p)` on a `UInt32` array gives `20`, not `5`), so the element count needs
`// size_of[Scalar[T]]()`; the first draft omitted that division. Verified with a real
`upper_bound` call over `[0,2,2,5,5,5]`, checking 4 boundary cases, not just a bare compile.

`HistoContainer.mojo`: composition (`var base: OneToManyAssoc[I, NHISTS*NBINS+1, SIZE]`), per the
C1 scoping decision. Forwards ~15 of `OneToManyAssoc`'s public methods under the same names
(`zero`, `add`, `initStorage`, `bulkFill`/`bulkFinalize`/`bulkFinalizeFill`, `finalize`,
`size`/`size(b)`, `begin`/`end`/`begin(b)`/`end(b)`) — except `count(b: Int)`/`fill(b, j)`, which
C++ itself doesn't expose through `HistoContainer` either: declaring `count(T)`/`fill(T,
index_type)` in the derived class hides all base overloads of those names from plain
`histo.count(...)` call sites (ordinary C++ name-hiding; no `using Base::count;` in the header),
so only the T-keyed overloads are reachable on a real `HistoContainer` — matched by simply not
forwarding the Int-keyed ones under the same names. `T` (the value being binned) is `T: DType`
with `Scalar[T]` values, not `OneToManyAssoc.I`'s `Copyable & Movable & ...` bound — `bin()` needs
real bit-shift/mask ops, which needs an actual `DType`. `NBINS`/`SIZE`/`S`/`NHISTS` are plain
`Int` comptime params (this port's own convention — `FlexiStorage.S`, `PrefixScan.block_size` —
not `mojo-serial`'s `UInt32`), with defaults matching C++'s (`S = size_of[Scalar[T]]()*8`,
`I = UInt32`, `NHISTS = 1`) — the first time this port has used comptime parameter defaults,
including one default expression referencing an earlier parameter (`S`'s default references `T`);
confirmed this works with no special handling needed. `bin()` needed `signed_to_unsigned[T]()`
(matching C++'s `std::make_unsigned<T>::type`, for a logical, non-sign-extending shift) — added to
`MojoBridge/DTypes.mojo`, carried over verbatim from `mojo-serial`'s own copy, extending the
existing "carried over, not present in mojocudatest" precedent already in that file.
`count(t)`/`fill(t,j)` call `Base.atomicIncrement`/`atomicDecrement` directly on
`self.base.off.data() + b`, not `self.base.count(b)`/`self.base.fill(b,j)` — matching C++ exactly
(`Base::atomicIncrement(this->off[b])`, not `Base::count(b)`), since `Base::count`'s own bound
check (`b < nOnes()`) is looser than what `HistoContainer` wants to assert here (`b < nbins()`).
Verified on real hardware: 10 `uint8` values distributed into `NBINS=4` bins via `bin()`'s
bit-shift hash, `zero()` → `count()` → `finalize()` → `fill()`, sizes and iteration both exact
(`size()=10`, per-bin `2/3/0/5`, matching a hand-computed top-2-bits-of-value split).

`forEachInBins`/`forEachInWindow` also ported (plain iteration helpers, no kernel launch
involved). One real, newly-discovered compiler limitation: passing a `capturing` closure as a
*runtime argument* doesn't work in this Mojo version ("TODO: capturing closures cannot be
materialized as runtime values" — a real compiler-emitted message, not inferred). Fixed by making
`func` a *comptime parameter* instead (confirmed working via `capturing_closure_check.mojo`) —
different from `mojo-serial`'s own signature shape (`func: fn (...) capturing -> None` as a
runtime arg), a toolchain-version difference, not a design choice. A related dialect quirk: this
version doesn't parse an inference-only `//` parameter marker followed by more explicit params in
the same bracket list (unlike `CUDAAtomics.mojo`'s `[dt: DType, //]`, which has nothing after
`//`) — so `func` can't be marked "explicit, after the inferred params" that way; calling
`forEachInBins[my_closure](...)` positionally matches `my_closure` to the *first* parameter
(`ValueT`) instead, which is wrong. Fixed by dropping `//` entirely and calling with `func=`
specified by keyword (`forEachInBins[func=my_closure](hist, ...)`), which correctly leaves
`ValueT..NHISTS` to ordinary inference from the `hist` argument. Verified on host (no GPU
dependency in either function) across 4 cases — single-bin, multi-bin via `n`, and two
`forEachInWindow` spans — all exact against hand-computed expected counts.

**Not ported: `countFromVector`, `fillFromVector`, `fillManyFromVector`, `launchZero`,
`launchFinalize`.** Explored and deliberately deferred, not blocked by what `launch.mojo`
documents. Re-examining that blocker precisely: it's specifically about forwarding an
already-bound kernel *value* through an opaque parameter (a fully generic `launch(any kernel,
...)` wrapper, matching C++'s `cms::cuda::launch`). That's different from what this group needs —
calling a *specific, named* generic kernel by name with type arguments, from inside another
generic function. Confirmed that pattern works fine, both via a trait-dispatch spike
(`kernel_trait_spike.mojo` — a generic function calling `ctx.compile_function` on a trait-required
static method, real hardware, correct output) and, more relevantly, via a plain generic kernel
called directly by name from another generic function with no trait at all
(`generic_kernel_from_generic_fn.mojo` — also real hardware, correct output). Neither
`countFromVector`/`fillFromVector` nor `launchZero`/`launchFinalize` need a caller-*pluggable*
kernel (they always call the same fixed kernels), so the simpler direct-call pattern is what
would actually apply here, not the trait one.

What actually blocks this group: `launchFinalize`'s device path needs `multiBlockPrefixScan`
(`totbins` can exceed 1024, so the existing single-block `blockPrefixScan` isn't enough), which
doesn't exist yet in this port — a separate, not-yet-scoped task. `countFromVector`/`fillFromVector`
are otherwise ready — `index_type` narrowed to `Scalar[IdxDType]` for a generic `IdxDType: DType`
lets the loop index convert to it via `Scalar[dt](i)` (confirmed working,
`scalar_dtype_construct_check.mojo`; a generic `Copyable & Movable & ...`-bound `I`, as
`OneToManyAssoc`/`HistoContainer` themselves use, has no such general from-`Int` construction —
tried a custom `FromInt`-style trait first, confirmed `UInt32`/`Int32` don't structurally satisfy
a self-defined trait without explicit conformance) — but they only make sense ported together
with the rest of this pipeline, not standalone. Deferred as one group, same as `OneToManyAssoc`'s
own launchers, pending `multiBlockPrefixScan`.

C1 (`OneToManyAssoc`/`HistoContainer` inversion) is otherwise complete: `FlexiStorage.mojo`,
`OneToManyAssoc.mojo`, `cudastdAlgorithm.mojo`, and `HistoContainer.mojo`'s core (composition,
forwarded methods, `bin`/`count`/`fill`, `forEachInBins`/`forEachInWindow`) are all written and
verified on real hardware. Remaining before C1 can be called fully done: `multiBlockPrefixScan` +
the five deferred launch functions above.

## Phase 2 (C3) — Track SoA restructure, started (2026-08-10)

Scoped before writing anything: `TrackSoAHeterogeneousT.h` (72 lines) and `TrajectoryStateSoAT.h`
(59 lines) both depend on `eigenSoA.h` (55 lines, `ScalarSoA`/`MatrixSoA` — Eigen-backed
struct-of-arrays storage) and `HeterogeneousSoA.h` (189 lines, the C4 target — a tagged
device/host/CPU unique-pointer wrapper). `mojo-serial` already has working ports of all four
(`EigenSoA.mojo`, `HeterogeneousSoA.mojo` -- though C4-incomplete, see its own section below --
`TrajectoryStateSoA.mojo`, `PixelTrackHeterogeneous.mojo`), so this is scoped as transcription +
dialect adaptation, same shape as `cudastdAlgorithm.mojo`, not new design -- except
`HeterogeneousSoA.mojo`, which needs real device-alloc work per C4.

Design question resolved before writing: `mojo-serial`'s `EigenSoA.mojo` uses `layout.LayoutTensor`
for matrix views, not `MojoCudaDev.MojoBridge.Matrix` (this port's own, independently-built Eigen
replacement, already used elsewhere). Confirmed these are meant to coexist rather than one
replacing the other: `Matrix.mojo` already has a `to_layout_tensor()` bridge function with a
comment explicitly documenting them as "two independently-ported Eigen equivalents" with different
memory layouts (row-major vs `Layout`'s col-major). So `eigenSoA.mojo` uses `LayoutTensor`,
matching `mojo-serial`, not a rewrite onto `Matrix`.

### Done — `CUDACore/eigenSoA.mojo`

Transcribed from `mojo-serial/CUDACore/EigenSoA.mojo`. First draft dropped two of three
`__init__` overloads on both `ScalarSoA` and `MatrixSoA` (the `InlineArray`-literal constructor
and the raw-pointer move/copy constructor) without any justification -- caught in review, not a
deliberate simplification, restored to match `mojo-serial` exactly.

Real dialect gaps found once actually compiled (this file's constructs hadn't been exercised
anywhere else in this port yet):
- Struct parameters (`T`, `S`, `R`, `C`) need explicit `Self.` qualification everywhere in this
  dialect, including in `comptime`/field declarations at the struct body level, not just inside
  method bodies -- `mojo-serial`'s `alias Scalar = Scalar[T]` needs `Scalar[Self.T]` here.
- `T.sizeof()` (a `DType` method in `mojo-serial`'s toolchain) doesn't exist in this one -- fixed
  via `size_of[Self.Scalar]()` (`from std.sys.info import size_of`), matching the convention
  `HistoContainer.mojo` already established for the same need.
- `__copyinit__`'s source parameter must be named `copy` in this dialect, not `other` — a plain
  compiler-enforced naming requirement, not seen before since no earlier file in this port needed
  `Copyable`.
- `layout`'s `IntTuple` needs importing explicitly (`mojo-serial` used it without importing it by
  name, presumably relying on a wildcard or older re-export that doesn't apply here).
- The origin-polymorphic `ref [origin]self -> UnsafePointer[X, mut=origin.mut, origin=origin]`
  pattern (used successfully elsewhere in this port, e.g. `VecArray.mojo`'s `begin()`/`end()`)
  fails with "inferred parameter passed out of order: 'mut'" specifically when the pointee type is
  `Scalar[dt]` rather than a plain opaque type parameter -- confirmed via three isolated spikes,
  narrowing it down precisely: reproduces even with every type fully concrete
  (`UnsafePointer[Scalar[DType.float32], mut=origin.mut, origin=origin]` inside a
  non-generic struct fails identically), so it's specifically about `Scalar[...]`, not genericity.
  A related origin-mismatch error (`UnsafePointer[Float32, origin_of(origin._data)]` vs
  `UnsafePointer[Float32, origin]`) appears if `mut=` is dropped instead. Not chased further to a
  root cause -- fixed pragmatically by dropping origin-polymorphism entirely on `ScalarSoA.data()`
  and `MatrixSoA.__getitem__`, using this port's own established simpler pattern instead
  (`mut self` + hardcoded `MutAnyOrigin`, as `OneToManyAssoc.mojo`/`FlexiStorage.mojo` already do)
  rather than `mojo-serial`'s more elaborate origin-polymorphic version. Since every real call site
  in this port so far only ever needs mutable access, this loses nothing in practice.

Verified via a real functional test (not just compile), matching this session's standing practice:
`ScalarSoA[DType.float32, 32]` — 32 sequential writes and reads, all exact, plus confirming
`.data()`'s pointer aliases `self[0]`. `MatrixSoA[DType.float32, 5, 1, 32]` — wrote and read back a
5-element column through the stride-based `[i]` indexing at two different indices, confirming they
don't alias each other (`index 0 unaffected by index 1 write`) and values match exactly at both.
Also a full project-wide cross-compile of `main.mojo` — clean.

### Done — `CUDADataFormats/HeterogeneousSoA.mojo` (C4, 2026-08-10)

Real device-side work, not transcription — `mojo-serial`'s version is `alias HeterogeneousSoA =
TypeableOwnedPointer`, a CPU-only stand-in with none of the device/host allocation machinery this
needs. C++ holds three mutually-exclusive `unique_ptr`s (`dm_ptr`/`hm_ptr`/`std_ptr`) and picks
whichever is non-null at each call — Mojo has no tagged-union type either, so this is the same
shape directly: three fields, `get()` checks them in order.

Scoped first: grepped all of `cudadev` for real construction call sites. `dm_ptr` (device) and
`hm_ptr` (host) both have genuine ones — the host path is constructed directly by the `*FromCUDA`
producers (e.g. `PixelTrackSoAFromCUDA.cc`), and the device path is asserted-and-read by
`toHostAsync()`, called on a device-side instance produced by not-yet-ported GPU kernel driver
code (Phase 4). `std_ptr` (plain CPU, no CUDA) has zero call sites anywhere in `cudadev` — kept
anyway for fidelity to the real class, at the user's request, modeled as a raw owning
`UnsafePointer` (manual alloc/destroy/free) rather than requiring `T: Defaultable`.

Two real gaps found once actually compiled and tested (this file's own dependencies hadn't been
exercised together before):
- `HostUniquePtr[T]` (`host_unique_ptr.unique_ptr[T]`, aliased to `OwnedPointer[_HostAllocation[T]]`)
  has no default constructor in this dialect, unlike C++'s `unique_ptr` — `OwnedPointer` always
  needs a value to wrap. Fixed via a small helper (`_null_host_ptr[T]()`) that wraps an explicitly
  default-constructed (internally null) `_HostAllocation[T]` instead of trying to default-construct
  the `OwnedPointer` itself.
- `UnsafePointer[T].alloc(1)` (the method-call form, used throughout `mojo-serial`, and referenced
  in an existing `CUDAAppContext.mojo` comment) doesn't exist in this dialect at all. The actual
  primitive is a free function, `alloc[T](count)` — confirmed via `ESProduct.mojo`'s own existing
  usage (`alloc[Item[Self.T]](1)`), not previously needed by anything std_ptr-shaped until now.

`toHostAsync()` takes an explicit `mut state: _AllocateHostState` parameter that C++'s doesn't need
(C++ reaches a global allocator implicitly) — matches the explicit-state pattern
`make_host_unique`/`make_device_unique` already established in this port, not a new deviation.
Its actual device→host copy reuses `copyAsync.mojo`'s existing single-element device-to-host
overload directly rather than reimplementing the memcpy.

Verified on real hardware, all four paths in one test: default construction (`get()` is null);
device path (a real kernel fills a device buffer, wrapped, `get()` non-null and correct);
`toHostAsync` (copies the device-filled buffer to a fresh host allocation, values match exactly --
111/222); host path (direct construction from `make_host_unique`, `get()` and `__getitem__` both
correct); std/raw path (manual `alloc[T]`-based ownership transfer, `get()` correct). Also a full
project-wide cross-compile of `main.mojo` — clean.

### Done — `CUDADataFormats/TrajectoryStateSoA.mojo` (2026-08-10)

Transcribed from `mojo-serial/CUDADataFormats/TrajectoryStateSoA.mojo`, matching `eigenSoA.mojo`'s
established fixes (`Self.` qualification, plain `Int` for `S` not `Int32`) plus several new
`LayoutTensor`-specific dialect gaps found here:

- Every `LayoutTensor[dt, layout]` type (comptime aliases, parameter types) needs an explicit third
  `origin` slot in this dialect — `_` to unbind it, matching `eigenSoA.mojo`'s general
  `Self.`-qualification finding but for a different parameter.
- `MatrixSoA` isn't `ImplicitlyCopyable` — `__copyinit__`'s field assignments need an explicit
  `.copy()` call (`self.state = copy.state.copy()`), not bare assignment.
- `copyToDense`'s C++ signature is `const`, but calling it needs `MatrixSoA.__getitem__`, which
  only has a `mut self` overload in this port (a real simplification from `eigenSoA.mojo`'s own
  origin-polymorphism dialect friction) — fixed by making `copyToDense`'s `self` `mut` too, a minor
  deviation from C++'s const-correctness that costs nothing here.
- `mojo-serial`'s `copyToDense` takes `v: LayoutTensor` fully unbound (no dtype/layout/origin at
  all) for its first parameter, unlike `cov`, which is dtype-bound. Initially over-specified `v`'s
  type by analogy with `cov` — wrong instinct, confirmed by testing: matching `mojo-serial`'s
  actual unbound form was necessary, not incidental. Even then, this dialect requires explicit
  proof the unbound tensor is mutable before allowing `__setitem__` on it (`v.origin.mut` "lacking
  evidence"), which `mojo-serial`'s compiler didn't need — resolved via `LayoutTensor[mut=True,
  ...]`, discovering along the way that this dialect's `...` (unbind-everything-remaining) syntax
  exists and works, `_` per-parameter doesn't scale past 2-3 unbound slots, and `mut=` as a keyword
  parameter must be textually first in the bracket list before any positional/unbound ones.
- Assigning a `.cast[CT]()`'d value from one `MatrixSoA`-sourced `LayoutTensor` into a differently
  laid-out one (`cov[j,j] = self.covariance[i][ind, 0].cast[CT]()`) fails to convert -- the cast
  result carries its *source* `LayoutTensor`'s `element_size` as part of its nominal type, not a
  clean `Scalar[CT]`, even though the underlying value is a plain scalar either way. Fixed with the
  same `rebind[Scalar[CT]](...)` pattern already needed for `v`'s assignment.

Verified via two real functional tests, not just compile: `copyFromDense`/`copyToDense` round-trip
(a 5-vector and a 5x5 symmetric matrix through the packed 15-element covariance storage) — every
element exact on read-back; `copyFromCircle` (the circle-fit parameterization, `state = [cp,
b*cp[2], lp]`, `cov` built from `ccov`/`lcov` with the `b`/`b²` scaling) — checked against
hand-computed expected values (`state = [1,2,6,4,5]`, `cov[0,0..2]=10,11,24`, `cov[2,2]=60`, etc.)
— all exact. Also a full project-wide cross-compile of `main.mojo` — clean.

### Done — `CUDADataFormats/TrackSoAHeterogeneousT.mojo` (2026-08-11)

Transcribed from `mojo-serial/CUDADataFormats/PixelTrackHeterogeneous.mojo` (which bundles
`TrackSoAT`/`PixelTrack`/the `PixelTrackHeterogeneous` alias into one file), but split back into
the two files C++ uses (`TrackSoAHeterogeneousT.h` and `PixelTrackHeterogeneous.h` are separate),
and corrected against the real C++ header rather than copied as-is:

- `hindex_type` is `uint32_t` in C++; `mojo-serial` had ported it as `uint16` — a real delta the
  port plan flagged up front, fixed here to `UInt32`.
- `GPU_SMALL_EVENTS` is a build-time `#ifdef` C++ never defines by default (`pixelTrack::maxNumber()`
  takes the `#else` branch, `32 * 1024`) — modeled as `comptime GPU_SMALL_EVENTS = False` plus a
  plain `if`/`else` in `maxNumber()`, the same idiom `is_gpu()` already uses elsewhere in this port
  for `#ifdef __CUDA_ARCH__`. First draft dropped this conditional entirely — caught in review.
- C++'s `pixelTrack::Quality` is a scoped enum (`enum class Quality : uint8_t`); `mojo-serial`
  flattened it to a bare `TrackQuality` struct with no connection to `PixelTrack`. First draft here
  copied that flattening onto `pixelTrack.bad`/`pixelTrack.dup`/etc. directly — caught in review
  ("quality should be pixeltrack quality"), restructured into a standalone `Quality` struct
  referenced as `pixelTrack.Quality = Quality`, giving proper `pixelTrack.Quality.bad`-style
  addressing that matches how C++ code actually spells it (`pixelTrack::Quality::bad`).
  `TrackSoAHeterogeneousT.Quality` is the storage type alias (`pixelTrack.Quality.T`, i.e. `UInt8`),
  kept separate from the `Quality` struct itself.

Two more dialect gaps, both matching patterns already established in `eigenSoA.mojo`/
`TrajectoryStateSoA.mojo`:
- `quality()`'s ref-return needs `ref [self.quality_._data]`, not `ref [self.quality_]` — must name
  `ScalarSoA`'s actual internal storage field, matching `ScalarSoA.__getitem__`'s own annotation.
- `charge()`/`phi()`/`tip()`/`zip()` all needed `rebind[Scalar[DType.float32]](...)` around the
  `.cast[DType.float32]()` reads from `stateAtBS.state`, the same `LayoutTensor`
  cross-element-size-type issue `TrajectoryStateSoA.mojo` already found.

One new, size-specific gap: `ScalarSoA[DType.uint8, S]`'s 128-byte alignment `constrained[]` needs
`S` itself to be a multiple of 128, not merely satisfying the `float32` fields (which only need a
multiple of 32) — surfaced as a real "function instantiation failed" error testing at `S=32`, fixed
by testing at `S=128` instead. `pixelTrack.TrackSoA` itself (`S = 32*1024`) is unaffected either way.

Verified via a real functional test at `S=128`: `stride()`; `quality(3)` reference-assignment
round-trip (`Quality.loose`, read back exact); `qualityData()` non-null; `phi`/`tip`/`zip`/`charge`
computed correctly off a real `copyFromDense` call into `stateAtBS` (expected `1.0/2.0/-5.0/1.0`,
all exact); `nHits(0) == 0` before any fills. Separately confirmed `pixelTrack.TrackSoA` itself
(`S=32768`, several MB) heap-allocates cleanly via `alloc[pixelTrack.TrackSoA](1)` with no crash.

### Found and fixed — `copyAsync.mojo` required `ImplicitlyCopyable`, but C++ never did (2026-08-11)

Finishing C3 meant checking that `HeterogeneousSoA[pixelTrack.TrackSoA]` (i.e.
`PixelTrackHeterogeneous`) actually satisfies `HeterogeneousSoA`'s type bound — it didn't compile.
Traced to `copyAsync.mojo` (Phase 1, written and verified before `HeterogeneousSoA.mojo` or
`TrackSoAHeterogeneousT.mojo` existed): its `_copy_kernel` did a typed, per-element `dst[i] =
src[i]` assignment, which silently requires `T: ImplicitlyCopyable`. C++'s
`HeterogeneousSoA::toHostAsync`/`copyAsync` never had that requirement — it's a raw
`cudaMemcpyAsync(dst, src, sizeof(T)*n, ...)`, a byte-level memcpy that works for any `T`, including
device-resident structs with non-copyable fields. `pixelTrack.TrackSoA` is exactly such a type — it
owns `OneToManyAssoc` fields (`hitIndices`/`detIndices`), which are `Movable` but not
`ImplicitlyCopyable`. This was a real, pre-existing semantic bug in the Phase 1 port, only surfaced
now because nothing had exercised `copyAsync` with a non-copyable-shaped `T` until this point.

Given the choice to defer (leave `PixelTrackHeterogeneous.mojo` unwritten until Phase 4 needs it
for real) or fix now, chose to fix now, at the user's explicit direction. Rewrote `_copy_kernel`/
`_launch_copy` to `bitcast[UInt8]()` both pointers and copy raw bytes, one thread per byte, with
grid size computed from `nelements * size_of[T]()` rather than `nelements` directly; relaxed
`_CopyElement` from `ImplicitlyCopyable & Movable & ImplicitlyDestructible` down to plain `AnyType`
(matching what `device_unique_ptr`/`host_unique_ptr` themselves require). All four `copyAsync[T]`
overloads keep their existing signatures.

Verified on real hardware: a plain `Int32` round-trip (regression check — still works after
switching to byte-copy); a 3-field `Multi` struct (`Movable`-only, deliberately shaped like a
non-copyable type) round-tripped byte-for-byte correctly; a 5-element `Int32` array multi-copy, all
exact. Confirmed via `git stash push -- copyAsync.mojo` / rebuild / `git stash pop` that a separate,
pre-existing hang in a *minimal* single-call test harness (fresh state, one `copyAsync` call)
reproduces identically on the original, unmodified file — not a regression from this fix, not
chased further. A fuller test that exercises `copyAsync` multiple times in sequence (matching this
fix's own verification, and later `HeterogeneousSoA`/`PixelTrackHeterogeneous` testing) works
correctly for all data, with only a late, unrelated segfault at final `DeviceContext` teardown
(inside `cuStreamIsCapturing`/`AsyncRT_DeviceContext_release`), after all useful work completes.

Relaxing `_CopyElement` broke `HeterogeneousSoA.mojo` in turn: its `__del__` calls
`self.std_ptr.destroy_pointee()`, which needs real destructibility that plain `AnyType` no longer
guarantees. Fixed by giving `HeterogeneousSoA.mojo` its own local bound,
`comptime _HeterogeneousElement = Movable & ImplicitlyDestructible`, instead of reusing
`copyAsync`'s now-looser one. Recompiling under the new bound also caught a second, independent bug
in `HeterogeneousSoA.__getitem__`: it returned `Self.T` by value (`return self.get()[]` with a
by-value return type), which silently required `ImplicitlyCopyable` too — a real design mismatch
with C++'s `operator*()`/`operator->()`, which both return references, never copies. Fixed to `fn
__getitem__(self) -> ref [self] Self.T`. Re-ran `HeterogeneousSoA.mojo`'s existing four-path
hardware test after both fixes — no regressions, all values still exact.

### Done — `CUDADataFormats/PixelTrackHeterogeneous.mojo` (2026-08-11) — C3 closed out

The simple part, once the above two fixes landed: `comptime PixelTrackHeterogeneous =
HeterogeneousSoA[pixelTrack.TrackSoA]`, matching C++'s one-line `using` alias exactly.

Verified on real hardware at full production scale (`pixelTrack.TrackSoA`, `S = 32*1024`, several
MB): device-allocated via `make_device_unique`, filled by a real kernel (`quality_[100] =
Quality.tight`, `chi2[100] = 3.5`), wrapped in `PixelTrackHeterogeneous`, copied back via
`toHostAsync` (exercising the new byte-copy kernel at real scale, not just on small test structs),
read back on the host — both values exact, no hang, no crash. Also ran a full project-wide
cross-compile of `main.mojo` (`--target-accelerator sm_80`) after all of this segment's changes —
clean, no errors.

C3 (Track SoA restructure) is now fully done: `eigenSoA.mojo`, `HeterogeneousSoA.mojo` (C4),
`TrajectoryStateSoA.mojo`, `TrackSoAHeterogeneousT.mojo`, `PixelTrackHeterogeneous.mojo` — all
ported, all verified on real hardware beyond just compiling.

## Phase 3 (D2/D3) — data formats and conditions, started (2026-08-11)

### Done — `CUDADataFormats/BeamSpotCUDA.mojo` (D2, 2026-08-11)

Small, direct port — C++'s `BeamSpotCUDA` is a 30-line wrapper around a single
`device::unique_ptr<BeamSpotPOD>`. `make_device_unique` here takes an explicit
`_AllocateDeviceState` param that C++'s constructor doesn't need (same deviation already used by
`HeterogeneousSoA.toHostAsync`/`BeamSpotCUDA`'s device_unique_ptr.mojo itself). `ptr()` needed a
reference-returning accessor (`ref [self.data_d_] DeviceUniquePtr[BeamSpotPOD]`), matching C++'s
`unique_ptr<T>& ptr()` — same shape as the `__getitem__` fix found earlier in `HeterogeneousSoA.mojo`.

Found, unrelated to this file itself: compiling it (which pulls in `DataFormats/BeamSpotPOD.mojo`,
a Bucket A file mechanically copied from `mojo-serial` early in the port but never actually
compiled, since nothing reachable from `main.mojo` used it yet) failed with `decorator
@register_passable("trivial") is removed, conform to TrivialRegisterPassable trait instead`. Fixed
by dropping the decorator and adding `TrivialRegisterPassable` to the trait list, matching the
convention already established elsewhere in this port (e.g. `DataFormats/DetId.mojo`). Four other
Bucket A files still carry the same removed decorator and are still unreached from `main.mojo`, so
still latently broken: `DataFormats/FEDTrailer.mojo`, `DataFormats/FEDHeader.mojo`,
`MojoBridge/Vector.mojo`, `DataFormats/SOARotation.mojo` — same one-line fix each, not yet applied
since nothing in this port calls them yet (per this port's standing rule: don't fix what's still
unreachable, but flag it for whoever reaches it first).

Verified on real hardware: default construction (`data()` null); the stream-allocating constructor
(`data()` non-null); a real kernel fill (`x/y/z/sigmaZ = 1.5/2.5/3.5/4.5`); read-back through
`ptr()` + `copyAsync` (mirroring `BeamSpotToCUDA.cc`'s own device→host pattern) — all four values
exact. Also a full project-wide cross-compile of `main.mojo` — clean.

### Done — `CUDADataFormats/SiPixelDigisCUDA.mojo` (D2, 2026-08-11)

Same shape as `BeamSpotCUDA.mojo` but larger: 7 device arrays (`xx_d`/`yy_d`/`adc_d`/`moduleInd_d`/
`clus_d`/`pdigi_d`/`rawIdArr_d`, all multi-element `device_unique_ptr`s sized by `maxFedWords`) plus
a `DeviceConstView` — a small struct of 5 raw device pointers, built host-side via
`make_host_unique`, filled in by direct field writes through the returned pointer, then pushed to
device with a single `copyAsync[DeviceConstView]` call. The constructor needed both an explicit
`_AllocateDeviceState` *and* `_AllocateHostState` param (not just one, like `BeamSpotCUDA.mojo`),
since staging `DeviceConstView` on the host before copying it to device needs both allocators live
at once — same deviation as before, just with twice the state.

One deliberate deviation from C++, confirmed correct rather than assumed: `mojo-serial`'s CPU-only
version rebuilds `view_d`'s internal pointers on every move (`__moveinit__`), because it held real
`List[T]` buffers whose backing storage could in principle move. This port's `view_d` instead wraps
a real device-resident buffer via `device_unique_ptr` — moving the Mojo-side struct only moves the
pointer *value*, never the GPU memory it points to, so the already-copied on-device `DeviceConstView`
stays valid without any rebuild. `__moveinit__` here is a plain field-by-field move, nothing special.

`@always_inline` was applied narrowly, not blanket-copied from `mojo-serial`'s style (which decorates
every method): only `DeviceConstView`'s 5 accessors (`xx`/`yy`/`adc`/`moduleInd`/`clus`) carry it,
matching that these are exactly the methods C++ itself marks `__forceinline__`
(`SiPixelDigisCUDA.h:53-58`) — `SiPixelDigisCUDA`'s own accessors have no such C++ annotation and
stay undecorated, consistent with `TrackSoAHeterogeneousT.mojo`/`BeamSpotCUDA.mojo`'s existing
no-blanket-decorator convention.

Verified on real hardware: `setNModulesDigis`/`nModules`/`nDigis` round-trip; `view()` non-null; a
real kernel wrote through `xx()`/`adc()`'s raw device pointers *and* read back through
`DeviceConstView.xx(i)`'s own device-side accessor (confirming the host-staged, `copyAsync`-copied
view genuinely points at the live device buffers, not stale/host addresses) — flagged the result
into `adc[4]`, read back via `adcToHostAsync`, exact (`adc[3]=30`, `adc[4]=99`). Also a full
project-wide cross-compile of `main.mojo` — clean.
