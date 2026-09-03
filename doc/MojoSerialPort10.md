# MojoSerial: porting `src/mojo-serial` from Mojo 25.5 to Mojo 1.0

Working notes and reference for the 25.5.0 → 1.0.0 migration on branch
`mojo-serial-1.0-port`. Everything here was verified against the 1.0 compiler on
this source or on an isolated probe — nothing is recalled from documentation.

> **STATE:** the tree is at **556 errors**. Both self-referential views are gone
> (`TrackingRecHit2DSOAView` §11, `ParamsOnGPU` §13) and the whole RecHits chain
> — `PixelCPEforGPU`, `PixelCPEFast`, `TrackingRecHit2DHeterogeneous`,
> `PixelRecHits`, `GPUPixelRecHits`, `SiPixelRecHitCUDA` — is at zero.
> Everything is uncommitted.

---

## 1. Toolchain

`pixi.toml` now pins:

```toml
mojo = "==1.0.0"
# provides `layout` (LayoutTensor); not part of the stdlib, so imported unprefixed
max  = "==26.5.0"
```

- `max 26.5.0` installs cleanly alongside `mojo 1.0.0` and supplies `layout.mojoc`.
- **`layout` is NOT under `std.`** — keep `from layout import Layout, LayoutTensor`.
- Version path from 25.5: 25.6 → 25.7 → 26.1 → 26.2 → 1.0 (five releases of drift).

Typecheck the whole package without needing a `main`:

```
pixi run mojo precompile MojoSerial -I . -o /tmp/x.mojoc
```

`precompile` rejects `main()` inside a package, so `bin/main.mojo` and the test
files each contribute one spurious "not supported within packages" error. Discount 6.

---

## 2. Language changes (Phase A — mechanical)

Applied by script, 2,665 edits across 117 of 127 files.

| Change | 25.5 | 1.0 | Sites |
|---|---|---|---|
| `fn` removed | `fn f()` | `def f()` | 1341 |
| …including function *types* | `fn (Int) capturing -> None` | `def (Int) capturing -> None` | 13 |
| compile-time binding | `alias K = 4` | `comptime K = 4` | 794 |
| compile-time control flow | `@parameter` + newline + `for`/`if` | `comptime for` / `comptime if` | 287 |
| stdlib moved under one package | `from sys import …` | `from std.sys import …` | 115 |
| size intrinsics snake_case | `sizeof`, `alignof` | `size_of`, `align_of` | 48 |
| trivial types become a trait | `@register_passable("trivial")` | `struct S(TrivialRegisterPassable)` | 30 |
| decorator removed outright | `@nonmaterializable(NoneType)` | *(deleted)* | 16 |
| origin intrinsic de-dundered | `__origin_of(x)` | `origin_of(x)` | 11 |
| type intrinsic de-dundered | `__type_of(x)` | `type_of(x)` | 1 |
| destructor renamed | `def __del__(var self)` | `def __deinit__(var self)` | 5 |
| `constrained` left the prelude | *(builtin)* | `from std.builtin import constrained` | 7 files |

**Semantics preserved.** Two properties verified before trusting the `fn`→`def`
sweep, both of which would have silently changed behaviour tree-wide:

- A plain 1.0 `def` **does not raise**. Calling a `raises` function from one fails
  with *"cannot call function that may raise in a context that cannot raise."*
- A plain 1.0 `def` argument **defaults to a read/borrow**, not by-value. Tested with
  an instrumented copy constructor; no copy fires.

`@parameter` on a **closure** is unchanged — only `@parameter for` / `@parameter if`
become `comptime`. The rewrite must distinguish them.

### Phase B — compiler fix-its

1.0 rejects unqualified access to a struct's own parameters and prints the exact
replacement. Scraping the diagnostics and applying them fixed **318** sites across
22 files in a single pass (a second pass found none).

```
error: unqualified access to struct parameter 'T'; use 'Self.T' instead
    comptime Scalar = Scalar[T]
                             ^
                             Self.T
```

---

## 3. Stdlib relocation map

Everything moved under `std.` except where noted.

| 25.5 | 1.0 |
|---|---|
| `sys`, `time`, `pathlib`, `memory`, `math`, `collections`, `bit`, `hashlib`, `utils`, `os`, `random`, `builtin`, `algorithm` | prefix with `std.` |
| `sys.ffi` | **`std.ffi`** (top level) |
| `os.atomic` | **`std.atomic`** (top level) |
| `compile.reflection` | **`std.reflection`** |
| `Span` | **`std.collections`** |
| `constrained` | `std.builtin` |
| `layout`, `LayoutTensor` | **unprefixed** — ships in MAX, not the stdlib |

Gone with no drop-in replacement:

- `algorithm.functional.parallelize` — not in stdlib *or* MAX. Use
  `std.runtime.asyncrt.TaskGroup` / `create_task`. One call site (`main.mojo:188`).
- `compile.reflection.get_type_name` — `std.reflection` has `get_linkage_name` only.
- `DType.sizeof()` — the method is removed; use the free function `size_of[DType.uint64]()`.
- `__str__` on scalars/DType — use `String(x)`.
- `Stringable`, `Representable` — no longer declared.

The stdlib also renamed every unsafe operation to carry an `unsafe_` prefix:
`bitcast`→`unsafe_bitcast`, `memset`→`unsafe_memset`,
`take_pointee`→`unsafe_take_pointee`, `steal_data`→`unsafe_take_allocation`,
pointer `+`/`+=`→`unsafe_offset`. Together with the `UnsafePointer` deprecation this
is a deliberate campaign to make unsafety lexically visible at every call site.

---

## 4. The pointer / origin model

**`UnsafePointer` is deprecated in 1.0.** The compiler says so directly:
`warning: 'UnsafePointer' is deprecated, use 'Pointer' instead`. Its `origin`
parameter is also now mandatory, so every pointer declaration has to be touched for
the 1.0 port regardless — the migration and the pointer refactor are the same job.

### Origins are compile-time only

| type | bytes |
|---|---|
| `Pointer[UInt16, origin_of(l)]` | 8 — same as a raw pointer |
| `Span` immutable | 16 — pointer + length |
| `Span[mut=True, …]` | 16 — *identical* |

Origins are type-level parameters in the `[]` list, erased before codegen. Borrow
checking costs nothing at runtime.

### The spellings that matter

```mojo
Span[T, _]                 # read-only view, origin unbound       (parameters)
Span[mut=True, T, _]       # writable view, no origin parameter   (parameters)
Span[T, origin_of(self.x_d[])].Immutable   # explicitly const     (returns)
Pointer[T, origin_of(self.bins)]           # single object / iterator
ref [self.field] T                         # reference to one object
```

Non-obvious facts, each verified:

- **`mut` on the binding does nothing for element writes.** `mut xx: Span[T, _]`
  makes `xx` itself reassignable; element mutability lives in the **origin**.
- **`Span[T, _]` is a real read-only contract.** Passing a *mutable* span to such a
  parameter still rejects writes — `_` binds `mut=False`, it does not inherit the
  caller's mutability. And the call still type-checks (mut → imm conversion).
- **`Span[mut=True, T, _]` keeps alias detection.** `_` unbinds the origin in the
  *signature*; the compiler still infers the concrete origin per call site:
  `error: aliasing values passed mutably to 'a' argument and passed mutably to 'b'`.
- **`.Immutable` requires a bound `mut`.** `Span[T, _].Immutable` is invalid
  (*"cannot access comptime member 'Immutable' with unbound parameter 'Span.mut'"*),
  so const-ness is spelled differently in parameters vs returns.
- **`Pointer` supports the whole C++ iterator idiom** — `+`, `+=`, `<`, `[]`, and
  `Int(ptr)` for addresses. Converting `begin()`/`end()` from `UnsafePointer` to
  `Pointer` needed no call-site changes at all.
- `MutAnyOrigin`, `MutUntrackedOrigin`, `ImmUntrackedOrigin` all exist.

### Borrow granularity — the rule that shapes every signature

Mojo checks borrows against the **signature**, not the body. A method taking
`ref self` declares a borrow of the *entire* struct, and is held to that even when
the body touches one field.

| form | borrows | several at once? |
|---|---|---|
| `soa.xx()` — `ref self` | the whole struct, exclusively | **no** |
| `soa.c_xx()` — `self` | the whole struct, shared | yes, and alongside field writes |
| `Span(soa.xx_d[])` | just the path `soa.xx_d` | yes — fields are disjoint |

Consequence: an SoA **cannot** expose usable mutable whole-array accessors.
`RawToDigi_kernel` takes six digis arrays in one call; six exclusive borrows of
`digis_d` cannot coexist. Field borrows can. This matches the pre-existing FIXME at
`GPUPixelRecHits.mojo:26` — *"not using views (passing a gazzilion of array
pointers) seems to produce the fastest code"*.

---

## 5. Conversion recipes

### SoA accessors

Three families, mirroring C++'s const/non-const pair plus the old view's element access:

```mojo
def xx(self, i: Int) -> UInt16:                                    # value; no borrow held
    return self.xx_d[][i]

def xx(ref self) -> Span[UInt16, origin_of(self.xx_d[])]:          # exclusive, one at a time
    return Span(self.xx_d[])

def c_xx(self) -> Span[UInt16, origin_of(self.xx_d[])].Immutable:  # shared, many at once
    return Span(self.xx_d[])
```

### Kernel signatures

```mojo
def RawToDigi_kernel(
    cablingMap: SiPixelFedCablingMapGPU,     # single object -> read borrow
    modToUnp: Span[UChar, _],                # read-only array
    xx: Span[mut=True, UInt16, _],           # writable array
    mut err: SimpleVector[...],              # container being pushed into
)
```

Call sites pass disjoint field borrows for the writable ones and keep `c_*` for the
read-only ones — this preserves the read/write intent and satisfies the checker:

```mojo
GPUClustering.findClus(
    self.digis_d.c_moduleInd(),              # read  — shared borrow
    self.clusters_d.c_moduleStart(),         # read
    Span(self.clusters_d.clusInModule_d[]),  # write — field borrow
    Span(self.digis_d.clus_d[]),             # write
)
```

### ES product accessors

```mojo
def getCPUProduct(self) -> ref [self.cablingMapHost] SiPixelFedCablingMapGPU:
    return self.cablingMapHost
```

Caller must bind with `ref`, not `var`. The types enforce this: these products are
`Movable` but **not** `Copyable`, so a `var` binding fails to compile rather than
silently deep-copying a cabling map.

### Binary search

Prefer Span + index over pointer pairs — it deletes the arithmetic entirely:

```mojo
def upper_bound[T: DType, //](s: Span[Scalar[T], _], var value: Scalar[T]) -> Int
```

```mojo
# before
var off = upper_bound(offsets, offsets + nh + 1, i)
var ih: Int32 = ((Int(off) - Int(offsets)) // size_of[UInt32]()) - 1
# after
var off = upper_bound(offsets[0 : Int(nh) + 1], UInt32(i))
var ih = Int32(off) - 1
```

### Self-referential views

A struct owning buffers *and* a view into them cannot name the origin — a field
cannot reference a sibling field's origin (`error: use of unknown declaration 'self'`).
Two outcomes:

1. **Delete the view** where it exists only for CUDA kernel marshalling. Both
   `DeviceConstView`s went this way; their hand-written `__moveinit__`, which existed
   solely to rebuild the view after a move, went with them.
2. **Untracked origin** where the view is genuinely needed across structs:
   `Span[T, ImmUntrackedOrigin]`, built with `rebind`. Lifetime is unchecked, but the
   element type and length survive — strictly better than a raw pointer.

---

## 6. Numeric strictness

1.0 **removed the implicit `UInt` → `Int` conversion** (it is not value-preserving).
This is the root of a large error class:

- `List(length=…)` takes `Int` — `List[T](length=UInt(n), fill=0)` fails.
- `range(a, b)` needs both `Int` — `range(UInt32, Int32)` has no overload.
- `min`/`max` will not mix `Int` with `Int32`.
- C++ relies on implicit narrowing that must now be spelled: `hist.fill(y[i], i - firstPixel)`
  becomes `hist.fill(y[i], UInt16(i - Int(firstPixel)))`.

**Fixing one conversion creates the next.** Changing `range(first, numElements)` to
`range(Int(first), Int(numElements))` made `i` an `Int`, which then broke
`min(msize, i)`. This category cannot be swept mechanically — each site needs the C++
checked to learn which type is authoritative.

---

## 7. Pitfalls and defects found

**Silent-corruption near-misses in the mechanical sweep** — both produced
plausible-looking source that the error count would not have caught:

- the `@register_passable` rule dropped a comma on the two structs whose conformance
  list spans multiple lines;
- the `@parameter for` rule swallowed a newline, leaving `comptime` orphaned on its
  own line.

Dry-run every rule against a copy and *read the diff*, not just the error count. A
blanket `sizeof`→`size_of` also corrupted a C++ cross-reference comment
(`HelixFitOnGPU.mojo:137`), which reverted; it was the only comment hit in 2,665 edits.

**Byte/element confusion.** C++ `h.end(b) - h.begin(b)` is pointer subtraction on
`IndexType*` and yields an *element* count. The port wrote
`Int(h.end(b)) - Int(h.begin(b))`, a *byte* difference — 4× too large with the default
`IndexType = uint32`, and the following `debug_assert(tot == rtot)` would fire. Five
sites in `test/HistoContainerTest.mojo`; every other pointer difference in the tree
divided correctly. Fixed with `h.size(b)`, which computes `off[b+1] - off[b]` directly
and removes the pointer arithmetic rather than correcting it.

**Compile-time tables indexed at runtime.** `ilog2`'s lookup tables were `alias` in
25.5 and indexed by a runtime loop variable — legal then, not in 1.0. Making them
`var` would lose their constness; the fix is `comptime for` plus extracting each
element as a comptime scalar, so they stay constants and never materialize:

```mojo
comptime for i in range(4, -1, -1):
    comptime bi = b[i]
    if v & bi: ...
```

**Dead debug code inflates the error count.** `comptime CPP_DUMP = False` in
`BrokenLine.mojo` gates 53 lines holding **34** of that file's 80 errors. Three
sibling flags (`RIEMANN_DEBUG`, `BROKENLINE_DEBUG`, `BL_DUMP_HITS`) are also
hardcoded false.

---

## 8. Reading the error count

**Fixing an upstream file raises the total**, because the compiler then reaches code
it previously could not parse. Clearing `HistoContainer` took that file 28 → 0 while
moving the total 706 → 751: `CAHitNtupletGeneratorKernels` went 50 → 80 and three
files began reporting errors for the first time. The same happened when `layout`
started resolving (793 → 802).

Use **files-at-zero**, not the total, as the progress metric.

---

## 9. Status

- 802 → **556** errors
- `UnsafePointer` 598 → **398**; `Span` 2 → **152**
- self-referential views 4 → **0** — none left in the port
- of 143 `.mojo` files, **81 still carry at least one error**
- 313 warnings remain, most of them the `UnsafePointer` deprecation — i.e. the
  remaining worklist restated

Fully converted: `SiPixelClustersSoA`, `SiPixelDigisSoA`, `HistoContainer`,
`CUDAStdAlgorithm`, `GPUCalibPixel`, `SiPixelFedCablingMapGPUWrapper`,
`SiPixelGainForHLTonGPU`, `SiPixelGainCalibrationForHLTGPU`, `GPUPixelRecHits`,
`SiPixelRecHitCUDA`. Plus `RawToDigi_kernel`, `countModules`, `findClus`,
`clusterChargeCut`, and the whole `cablingMap` thread.

## 10. Remaining work

| Item | Errors | Notes |
|---|---|---|
| `CUDACompat` | 7 | **next up** — tiny, all-`@deprecated` no-op shims, but it sits in the hit SoA's import closure and blocks building that code to assembly |
| `CAHitNtupletGeneratorKernels` | 161 | holds 25 of the 50 iterator sites; largely still raw-translated |
| `BrokenLine` | 84 | 34 are dead `CPP_DUMP` code — decide fix vs delete |
| `OrderedMap` / `OrderedMultiSet` | 78 | container shims |
| 50 `begin()`/`end()` iterator sites | — | Span conversion; `len(span)` makes the byte/element bug unrepresentable |
| `Int`/`UInt32` conversions | many | not mechanically sweepable, see §6 |

---

## 11. DONE — removing `TrackingRecHit2DSOAView`

**Complete.** 722 → 675 errors across the whole change. No view type remains in
the port; `TrackingRecHit2DSOAView.mojo` is deleted.

### Decision taken

Three options were weighed; the chosen one was **delete the view and unpack the
stores**, accepting layout divergence from `src/serial` in exchange for no views
and no untracked origins anywhere.

Rejected alternatives, for the record:

- *Convert the view in place* (37 pointer fields → `Span[mut=True, T, MutUntrackedOrigin]`).
  Keeps the C++ layout exactly, but leaves one untracked-origin view standing.
  This shape was verified working, including the punned column.
- *Delete the view, accessors on the owner, keep packed stores.* Punned-column
  **writes** need a pointer-level bitcast to yield a mutable reference, so raw
  pointers come back at ~4 sites.

### Done

- `TrackingRecHit2DHeterogeneous.mojo` rewritten: 13 typed column buffers, the
  view's accessors folded on, `m_view` gone.
- `TrackingRecHit2DSOAView.mojo` **deleted**. `Hist` (nested in the struct in
  C++) now lives in `TrackingRecHit2DHeterogeneous.mojo`; the one consumer that
  imported it, `gpuPixelDoubletsAlgo.mojo`, imports it from there.
- All nine consumers repointed. Four were pure alias retargets
  (`HitsOnGPU`/`HitsView`/`Hits` → `TrackingRecHit2DHeterogeneous`); the rest
  had `UnsafePointer[View]` parameters, all of which became plain borrows.
- `GPUPixelRecHits.getHits` converted whole: five pointer parameters
  (`cpeParams`, `bs`, `pdigis`, `pclusters`, `phits`) became borrows, with
  `mut hits` the only mutable one. This file carried all three deleted views
  (digis, clusters, hits) and is now at **0 errors**, as is
  `SiPixelRecHitCUDA.mojo`.

### Column mapping (old packed → new unpacked)

Old wire-up used two closures local to the constructor,
`get32(i) = m_store32.unsafe_ptr() + i * nHits` and the same over `m_store16` —
"start of column `i` at stride `nHits`". They existed only to wire the view's
pointers and have no replacement.

| old | new field |
|---|---|
| `get32(0..7)` | `m_xl_d`, `m_yl_d`, `m_xerr_d`, `m_yerr_d`, `m_xg_d`, `m_yg_d`, `m_zg_d`, `m_rg_d` — `List[Float]` |
| `get32(8).bitcast[Int32]()` | `m_charge_d: List[Int32]` |
| `get32(9)`, the `+11` tail | `m_hitsLayerStart_d: List[UInt32]`, length `numberOfLayers + 1` |
| `get16(0).bitcast[Int16]()` | `m_iphi_d: List[Int16]` |
| `get16(1)` | `m_detInd_d: List[UInt16]` |
| `get16(2).bitcast[Int16]()` | `m_xsize_d: List[Int16]` |
| `get16(3).bitcast[Int16]()` | `m_ysize_d: List[Int16]` |

Five `bitcast`s eliminated; the C++ `static_assert(sizeof(uint32_t) == sizeof(float))`
that licensed the punning is no longer load-bearing.

Fields that collapsed — the old design carried each supporting object twice, once
on the owner and once on the view: `m_hist`, `m_iphi`, `m_hitsLayerStart`,
`m_averageGeometry`. Also gone: `m_view`, `comptime n16`/`n32`, and the
hand-written `__moveinit__` (now synthesizable).

Two forwarded (non-owned) fields changed representation rather than being unpacked:

- `m_cpeParams: UnsafePointer[ParamsOnGPU]` → **`ParamsOnGPU` by value.**
  Measured 32 bytes — it is four pointers and nothing else, so copying it copies
  addresses, not geometry. Note its own four fields are still `UnsafePointer`;
  `PixelCPEforGPU.mojo` is a separate job and currently does not compile.
- `m_hitsModuleStart: UnsafePointer[UInt32]` → **copied into an owned
  `List[UInt32]`** (~8 KB/event). Equivalent because `clusters_d` is not mutated
  after this point, but it is a copy that did not exist before.

### Cost, corrected

An earlier note in this session said "14 allocations instead of 2" — that was
wrong, conflating column buffers with total allocations. Actual per-event
`OwnedPointer` count:

| | old | new |
|---|---|---|
| column storage | 2 | 13 |
| Hist / AverageGeometry | 2 | 2 |
| the view itself | 1 | 0 |
| hitsLayerStart | 0 (lived in the `m_store32` tail) | 1 |
| hitsModuleStart | 0 (borrowed) | 1 |
| **total** | **5** | **17** |

This is a steeper cost than was quoted when the option was chosen. It is a
one-allocation-per-column consequence of unpacking and is not recoverable
without repacking the stores; noted here so the tradeoff stays visible.

### `ref` returns: the origin must name the exact expression

Found while fixing the 13 folded accessors. Mojo 1.0 accepts **no widening** of
a returned reference's origin — it must name the exact access path:

```mojo
def x(ref self, i: Int) -> ref [self.m_xl_d[]] Float:      # ✗ names the List
def x(ref self, i: Int) -> ref [self] Float:               # ✗ widening to self
def x(ref self, i: Int) -> ref [self.m_xl_d[][i]] Float:   # ✓ names the element
```

The error reads `cannot return reference with incompatible origin:
'origin_of(self.m_xl_d["value"]["element"])' vs 'origin_of(self.m_xl_d["value"])'`
— `["element"]` is the tell that the declared origin stopped one level short.
Widening a plain field (`ref [self]` returning `self.p`) is rejected the same
way; `ref [self.p]` is the accepted form.

### Verification

Per-file counts after the change: `GPUPixelRecHits` and `SiPixelRecHitCUDA` at
**0**; `TrackingRecHit2DHeterogeneous` at **2**, both cascades from
`ParamsOnGPU` in the still-un-ported `PixelCPEforGPU.mojo` (it is not `Movable`
and its field origin degrades to `MutUnsafeAnyOrigin`). Every residual error in
the other seven consumers is the generic
`UnsafePointer[T]`-without-origin category, pre-existing and unrelated.

**`PixelCPEforGPU.mojo` is therefore the next file** — it is the only thing
still holding the hit SoA back.

---

## 12. Codegen — measured, not assumed

All figures below are from `mojo build --emit asm`, default `-O3`, x86-64.
Probes live in the scratchpad (`asm/layout.mojo`, `asm/spans.mojo`,
`asm/thirteen.mojo`).

### Benchmark with `-D ASSERT=none` or you measure bounds checks

With assertions on (the default) the same loop is **207 lines** of assembly
instead of **31** — `String::write_to`, `_debug_assert_msg` and `_printf` cold
paths for every `List.__getitem__`. Any codegen comparison without
`-D ASSERT=none` is measuring the assertion machinery, not the code.

### Per-element accessors defeat the backend

An `@always_inline` accessor returning `ref [self.m_xl_d[][i]]` inlines fine,
but the `OwnedPointer -> List -> data` indirection is **reloaded on every
iteration**. A store through any column may alias the pointer cells holding the
other columns' base addresses, so LLVM cannot hoist them.

Loop reading 3 columns and writing a 4th:

| form | loop body | pointer reloads / iter |
|---|---|---|
| accessors on the unpacked struct | 14 instr | **4** |
| packed store (the C++ `m_store32` layout) | 12 instr | 1 |
| Spans bound before the loop | **9 instr** | **0** |
| Spans as parameters | **9 instr** | **0** |

At the real column count (13, `store_accessors` vs `store_spans`) the gap
widens rather than closing — register pressure was the worry, and it did not
materialise:

| form | loop body | pointer reloads / iter |
|---|---|---|
| accessors | 33 instr | **13** |
| Spans bound before the loop | **18 instr** | **0** |

All 13 base addresses stay in registers; only two loop-invariant scalars spill.

**Consequence: binding columns to Spans once, before the loop, beats even the
packed C++ layout** — packed still reloads its single base pointer and computes
column offsets. The unpacking decision (§11) therefore costs nothing at the
access site, provided hot loops bind Spans instead of calling per-element
accessors. Applied to `GPUPixelRecHits.getHits`; the 50 iterator sites in §10
are the same pattern.

This also vindicates the upstream C++ FIXME in `getHits` ("not using views —
passing a gazzilion of array pointers — seems to produce the fastest code, but
it is harder to maintain"). A Span *is* that array pointer, with the bounds
carried along, so the maintainability objection does not transfer.

Note the accessors must stay on the struct regardless — they are the API for
scalar, non-loop access. The Span binding is a hot-loop technique, not a
replacement.

### Runtime-indexing a `comptime` table: `materialize`

A `comptime InlineArray` indexed by a **runtime** value fails with *"cannot
materialize comptime value of type `Array[Float32, N]` to runtime because it is
not `ImplicitlyCopyable`"*. The table does **not** have to become a `var` — the
compiler's own fix-it is `materialize`:

```mojo
comptime tbl: InlineArray[Float, 3] = [0.00115, 0.00120, 0.00088]
...
materialize[tbl]()[Int(sx)] if Int(sx) < comptime (len(tbl)) else tbl_def
```

`materialize` is a builtin — do **not** import it (`std.builtin` has no such
name). Two further notes: `len()` on a comptime table materialises it too, so it
needs `comptime (len(tbl))`; and a `comptime` **SIMD** vector indexes by a
runtime value with no ceremony at all, so it is the lighter choice for a table
that is only ever read.

Applied to the six error tables in `errorFromSize`/`errorFromDB`.

### Origins do not become `noalias`

Nothing vectorises — every variant stays scalar (`vmovss`/`vfmadd231ss`). This
is **not** the FP reduction: a pure elementwise loop over three Spans with
distinct origins and no accumulator is still scalar, and LLVM does not even
emit its usual runtime-overlap-check-plus-vector-loop pair.

So origins are erased before codegen. §4 records that they are compile-time
only and cost nothing at runtime; the corollary is that they also **buy**
nothing — they are a borrow-checking device, not an aliasing hint. Do not
expect a `Span[T, _]` parameter to imply `noalias` to the backend.

---

## 13. DONE — removing `ParamsOnGPU`, the second self-referential view

**Correction to §11's claim that no views remained.** `PixelCPEFast._cpuData`
was one: four `UnsafePointer`s into `PixelCPEFast`'s *own* fields, rebuilt in
both constructors and in `__moveinit__`. `ParamsOnGPU` was its view type. The
count was 4 → 1, not 4 → 0. It is now genuinely 0.

`PixelCPEFast` was also broken independently: `__moveinit__` does not exist in
1.0, so the rebuild-the-pointers-after-move hook that made `_cpuData` sound was
not running at all.

### What changed

`ParamsOnGPU` is deleted. Its five accessors — `commonParams()`, `detParams(i)`,
`layerGeometry()`, `averageGeometry()`, `layer(id)` — moved onto `PixelCPEFast`,
returning refs into its own fields. `_cpuData`, `getCPUProduct()` and the
hand-written move constructor are gone; the move is synthesized.

Consumers take `PixelCPEFast` as a borrow. The chain that changed:

| site | before | after |
|---|---|---|
| `SiPixelRecHitCUDA` | `UnsafePointer(to=es.get[PixelCPEFast]().getCPUProduct())` | `es.get[PixelCPEFast]()` |
| `PixelRecHitGPUKernel.makeHits` | `var cpeParams: UnsafePointer[ParamsOnGPU]` | `cpeParams: PixelCPEFast` |
| `setHitsLayerStart` | three `UnsafePointer[…]` | two `Span`s + `PixelCPEFast` |
| `getHits` | `cpeParams: ParamsOnGPU` | `cpeParams: PixelCPEFast` |
| `TrackingRecHit2DHeterogeneous` | stored `m_cpeParams` by value | field dropped entirely |

Dropping `m_cpeParams` is the notable one: the hit SoA stored it only to serve
two call sites, both in still-un-ported files. Those two now need `cpeParams`
threaded in as a parameter:

```
plugin_PixelTriplets/RiemannFitOnGPU.mojo:73
plugin_PixelTriplets/BrokenLineFitOnGPU.mojo:81
    hhp[].cpeParams().detParams(Int32(hhp[].detectorIndex(hit))).frame.toGlobal(
```

`detParams` now takes `Int` rather than `Int32`, matching every other index
accessor in the port.

### Result

675 → **556** errors. All six files in the RecHits chain are at zero:
`PixelCPEforGPU`, `PixelCPEFast`, `TrackingRecHit2DHeterogeneous`,
`PixelRecHits`, `GPUPixelRecHits`, `SiPixelRecHitCUDA`.

Most of the residue was §6 numeric strictness, arriving exactly as described
there — one conversion fixed uncovers the next (`range` → `__lt__` → `__ne__` →
`__add__` → parameter type), five rounds before it settled.

### Still blocked: assembly for the real `getHits`

`CUDACompat.mojo` (7 errors) sits in the hit SoA's import closure, so the §12
Span change is verified on a structural probe but not yet on the real function.
The errors are two shapes only — `OpaquePointer()` with no null spelling, and
`UnsafePointer[Scalar[T], mut=True]` where `mut` is no longer a keyword
parameter — in a struct that is entirely `@deprecated` no-op shims for the
serial backend. Fixing it is the cheapest path to a real before/after.