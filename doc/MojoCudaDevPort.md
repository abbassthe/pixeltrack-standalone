# Porting `cudadev` to Mojo — Work Plan

**Target:** `src/mojo-cudadev/` — a Mojo port of `src/cudadev`, the development CUDA
version of the pixel-tracking reconstruction, running real GPU kernels via
`std.gpu.host.DeviceContext`.

**Status:** Phase 0 started.
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

- `MojoBridge/` is still **incomplete**. The merge in §6 needs six modules that exist
  only in `mojo-serial` and are in the older dialect:
  `Matrix.mojo` (1743 lines), `Vector.mojo` (925), `SymmetricEigen.mojo` (324),
  `Print.mojo` (105), `Timer.mojo` (124), `Static.mojo` (63 — contains one of the 11
  `.alloc(` sites). ~3.3k lines total.
- Everything else sourced from `mojo-serial` — i.e. every algorithm.

`MojoBridge/Print.mojo`, `Timer.mojo` and `Static.mojo` (292 lines combined, few
dependencies, and `Static.mojo` exercises the `.alloc(` change) are a **better
Phase 0a spike subject than the Framework files** originally proposed in §7. Use
those three.
