# Mojo port of CUDACore/HistoContainer.h.
#
# No inheritance in Mojo: composition instead of
# `class HistoContainer : public OneToManyAssoc<I, NHISTS*NBINS+1, SIZE>`.
# HistoContainer holds `base: OneToManyAssoc[...]` and forwards its public
# methods explicitly -- except `count(b: Int)`/`fill(b, j)`, which C++ itself
# doesn't expose through `HistoContainer` either: declaring `count(T)`/
# `fill(T, index_type)` in the derived class hides all base overloads of
# those names from plain `histo.count(...)` call sites (C++ name-hiding
# rules; there's no `using Base::count;` in the header), so only the T-keyed
# overloads are reachable on a real HistoContainer -- matched here by simply
# not forwarding the Int-keyed ones under the same names.
#
# HistoContainer's own count(t)/fill(t,j) call Base's atomicIncrement/
# atomicDecrement directly on `self.base.off.data() + b`, not `self.base.count(b)`
# -- matching C++ exactly (`Base::atomicIncrement(this->off[b])`, not
# `Base::count(b)`), since Base::count's own bound check (`b < nOnes()`) is
# looser than what HistoContainer wants to assert here (`b < nbins()`).
#
# Not ported: countFromVector, fillFromVector, fillManyFromVector, launchZero,
# launchFinalize. Not the same "generic-kernel-launch problem" as launch.mojo
# claims for the general case -- confirmed that calling a specific, named
# generic kernel by name with type arguments, from inside another generic
# function, actually works fine (kernel_trait_spike.mojo,
# generic_kernel_from_generic_fn.mojo). What actually blocks this group:
# launchFinalize's device path needs multiBlockPrefixScan (bins can exceed
# 1024, so the existing single-block blockPrefixScan isn't enough), which
# doesn't exist yet -- a separate task. countFromVector/fillFromVector are
# otherwise ready (index_type narrowed to Scalar[IdxDType] for a generic
# IdxDType: DType, so the loop index converts to it via Scalar[dt](i) --
# confirmed working, see scalar_dtype_construct_check.mojo) but only make
# sense ported together with the rest of this pipeline.
from std.sys.info import size_of
from std.memory import AddressSpace

from MojoCudaDev.CUDACore.OneToManyAssoc import OneToManyAssoc, OneToManyAssocView
from MojoCudaDev.CUDACore.AtomicPairCounter import AtomicPairCounter
from MojoCudaDev.MojoBridge.DTypes import signed_to_unsigned


struct HistoContainer[
    T: DType,  # the type of the discretized input values
    NBINS: Int,  # number of bins
    SIZE: Int,  # max number of elements; -1 for runtime-sized external storage
    S: Int = size_of[Scalar[T]]() * 8,  # number of significant bits in T
    I: Copyable & Movable & ImplicitlyCopyable & Defaultable = UInt32,  # type stored (usually an index)
    NHISTS: Int = 1,  # number of histos stored
](Movable):
    comptime Base = OneToManyAssoc[Self.I, Self.NHISTS * Self.NBINS + 1, Self.SIZE]
    comptime View = Self.Base.View
    comptime Counter = Self.Base.Counter
    comptime CountersOnly = Self.Base.CountersOnly
    comptime index_type = Self.Base.index_type
    comptime UT = signed_to_unsigned[Self.T]()

    var base: Self.Base

    fn __init__(out self):
        self.base = Self.Base()

    fn __moveinit__(out self, deinit take: Self):
        self.base = take.base^

    @staticmethod
    fn ilog2(x_in: UInt32) -> UInt32:
        return Self.Base.ilog2(x_in)

    @staticmethod
    fn sizeT() -> Int:
        return Self.S

    @staticmethod
    fn nbins() -> Int:
        return Self.NBINS

    @staticmethod
    fn nhists() -> Int:
        return Self.NHISTS

    @staticmethod
    fn totbins() -> Int:
        return Self.NHISTS * Self.NBINS + 1

    @staticmethod
    fn nbits() -> UInt32:
        return Self.ilog2(UInt32(Self.NBINS - 1)) + 1

    @staticmethod
    fn histOff(nh: UInt32) -> UInt32:
        return UInt32(Self.NBINS) * nh

    @staticmethod
    fn bin(t: Scalar[Self.T]) -> Scalar[Self.UT]:
        var shift = UInt32(Self.sizeT()) - Self.nbits()
        var mask = (UInt32(1) << Self.nbits()) - 1
        return ((t.cast[DType.uint32]() >> shift) & mask).cast[Self.UT]()

    # ---- forwarded from Base ----

    fn initStorage(mut self, view: Self.View):
        self.base.initStorage(view)

    fn zero(mut self):
        self.base.zero()

    fn add(mut self, read co: Self.CountersOnly):
        self.base.add(co)

    fn bulkFill(
        mut self,
        mut apc: AtomicPairCounter,
        v: UnsafePointer[Self.index_type, MutAnyOrigin],
        n: UInt32,
    ) -> Int32:
        return self.base.bulkFill(apc, v, n)

    fn bulkFinalize(mut self, read apc: AtomicPairCounter):
        self.base.bulkFinalize(apc)

    fn bulkFinalizeFill(mut self, read apc: AtomicPairCounter):
        self.base.bulkFinalizeFill(apc)

    fn finalize[
        block_size: Int, ws_address_space: AddressSpace
    ](mut self, ws: UnsafePointer[Self.Counter, MutAnyOrigin, address_space=ws_address_space]):
        self.base.finalize[block_size=block_size](ws)

    fn size(mut self) -> UInt32:
        return self.base.size()

    fn size(mut self, b: Int) -> UInt32:
        return self.base.size(b)

    fn begin(mut self) -> UnsafePointer[Self.index_type, MutAnyOrigin]:
        return self.base.begin()

    fn end(mut self) -> UnsafePointer[Self.index_type, MutAnyOrigin]:
        return self.base.end()

    fn begin(mut self, b: Int) -> UnsafePointer[Self.index_type, MutAnyOrigin]:
        return self.base.begin(b)

    fn end(mut self, b: Int) -> UnsafePointer[Self.index_type, MutAnyOrigin]:
        return self.base.end(b)

    # ---- HistoContainer's own value-keyed count/fill ----

    fn count(mut self, t: Scalar[Self.T]):
        var b = Int(Self.bin(t))
        debug_assert(b < Self.nbins(), "HistoContainer.count: bin out of range")
        _ = Self.Base.atomicIncrement(self.base.off.data() + b)

    fn fill(mut self, t: Scalar[Self.T], j: Self.index_type):
        var b = Int(Self.bin(t))
        debug_assert(b < Self.nbins(), "HistoContainer.fill: bin out of range")
        var w = Self.Base.atomicDecrement(self.base.off.data() + b)
        debug_assert(w > 0, "HistoContainer.fill: bin already empty")
        self.base.content[Int(w) - 1] = j

    fn count(mut self, t: Scalar[Self.T], nh: UInt32):
        var b = Int(Self.bin(t)) + Int(Self.histOff(nh))
        debug_assert(b < Self.totbins(), "HistoContainer.count: bin out of range")
        _ = Self.Base.atomicIncrement(self.base.off.data() + b)

    fn fill(mut self, t: Scalar[Self.T], j: Self.index_type, nh: UInt32):
        var b = Int(Self.bin(t)) + Int(Self.histOff(nh))
        debug_assert(b < Self.totbins(), "HistoContainer.fill: bin out of range")
        var w = Self.Base.atomicDecrement(self.base.off.data() + b)
        debug_assert(w > 0, "HistoContainer.fill: bin already empty")
        self.base.content[Int(w) - 1] = j


# C++: forEachInBins -- HistoContainer.h:72-81
#
# func is a comptime parameter, not a runtime argument: passing a `capturing`
# closure as a runtime value isn't supported in this Mojo version ("TODO:
# capturing closures cannot be materialized as runtime values" -- confirmed
# directly, see capturing_closure_check.mojo, which also confirms a comptime
# parameter works fine). Call as `forEachInBins[func=my_closure](hist, ...)`,
# not positionally (`forEachInBins[my_closure](...)`) -- the inferred params
# (ValueT..NHISTS) can't be marked infer-only via `//` here since this
# dialect doesn't parse explicit params after a `//` marker in the same
# list, so plain positional bracket args match from the first parameter,
# not skipping the inferred ones; keyword `func=` sidesteps that.
fn forEachInBins[
    ValueT: DType, NBINS: Int, SIZE: Int, S: Int,
    IdxT: Copyable & Movable & ImplicitlyCopyable & Defaultable, NHISTS: Int,
    func: fn (IdxT) capturing -> None,
](
    mut hist: HistoContainer[ValueT, NBINS, SIZE, S, IdxT, NHISTS],
    value: Scalar[ValueT],
    n: Int,
):
    var bs = Int(hist.bin(value))
    var be = min(NBINS - 1, bs + n)
    bs = max(0, bs - n)
    debug_assert(be >= bs, "forEachInBins: be must be >= bs")
    var pj = hist.begin(bs)
    var last = hist.end(be)
    while Int(pj) != Int(last):
        func(pj[])
        pj += 1


# C++: forEachInWindow -- HistoContainer.h:84-92
fn forEachInWindow[
    ValueT: DType, NBINS: Int, SIZE: Int, S: Int,
    IdxT: Copyable & Movable & ImplicitlyCopyable & Defaultable, NHISTS: Int,
    func: fn (IdxT) capturing -> None,
](
    mut hist: HistoContainer[ValueT, NBINS, SIZE, S, IdxT, NHISTS],
    wmin: Scalar[ValueT],
    wmax: Scalar[ValueT],
):
    var bs = Int(hist.bin(wmin))
    var be = Int(hist.bin(wmax))
    debug_assert(be >= bs, "forEachInWindow: be must be >= bs")
    var pj = hist.begin(bs)
    var last = hist.end(be)
    while Int(pj) != Int(last):
        func(pj[])
        pj += 1
