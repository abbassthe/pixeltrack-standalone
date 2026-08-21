# Mojo port of CUDACore/OneToManyAssoc.h.
#
# The free-function kernel launchers (`zeroAndInit`, `launchZero`,
# `launchFinalize`, `finalizeBulk`) are not ported -- same generic-kernel-
# launch problem as launch.mojo.
#
# View is generic over Assoc bound to a trait declaring Counter/index_type,
# matching C++'s `using Counter = typename Assoc::Counter` shape -- but
# never actually *projects* through it (no `Self.Assoc.Counter`/`.index_type`
# anywhere): offStorage stays hardcoded UnsafePointer[UInt32, ...] (Counter
# never varies), and contentStorage stays an untyped pointer, bitcast to the
# real element type only where it's actually used, inside OneToManyAssoc's
# own initStorage (where Self.index_type is a plain leaf parameter, not a
# projection through Assoc). Actually projecting through Self.Assoc.X on a
# self-referential Assoc=Self crashes this compiler with infinite recursion
# -- confirmed still true even after Bug A's FlexiStorage fix (see
# selfref_view_from_assoc.mojo); merely having the trait bound present,
# unused, does not crash (see selfref_view_trait_unused.mojo).
#
# Tried (and reverted) an alternative: making initStorage independently
# generic over its own Assoc parameter (never writing "Self" in its own
# signature/body) with contentStorage properly typed as Assoc.index_type.
# That compiles fine standalone (no calls) but crashes identically to Bug B
# once actually *called* from anywhere else -- standalone compilation of a
# generic method's definition doesn't fully monomorphize it, a real call
# site is needed to trigger the crash. Also tried removing FlexiStorage's
# `@parameter if` (Bug A's fix) as a possible additional factor for this
# specific crash -- confirmed it isn't; the call-site crash persists either
# way, so this is purely Bug B, unrelated to Bug A.
#
# setContentStorage() below gets real type-checking back on the write side
# despite contentStorage being opaque: it's a method on View itself (not on
# the self-referential OneToManyAssoc), so requiring exactly
# Self.Assoc.index_type as its parameter type doesn't trigger Bug B -- View
# isn't the self-referential struct, only OneToManyAssoc is, and this method
# never gets called from inside OneToManyAssoc's own elaboration. Confirmed
# via a real call site, not just a standalone definition (see
# selfref_view_typed_setter.mojo). Without this, a caller could bitcast any
# type at all into contentStorage with no static check whatsoever.
#
# atomicIncrement/atomicDecrement take a pointer instead of a `Counter&`:
# FlexiStorage's __getitem__ returns a value, not a reference, so there's no
# Mojo equivalent of `off[b]` as an lvalue to hand to an atomic op -- callers
# get a pointer via `.data() + b` instead.
from std.memory import AddressSpace

from MojoCudaDev.CUDACore.FlexiStorage import FlexiStorage
from MojoCudaDev.CUDACore.CUDAAtomics import atomic_fetch_add, atomic_fetch_sub
from MojoCudaDev.CUDACore.PrefixScan import blockPrefixScan
from MojoCudaDev.CUDACore.AtomicPairCounter import AtomicPairCounter

# C++: template <typename Assoc> struct OneToManyAssocView -- OneToManyAssoc.h:22-32
trait OneToManyAssocLike:
    comptime Counter: Copyable & Movable & ImplicitlyCopyable & Defaultable
    comptime index_type: Copyable & Movable & ImplicitlyCopyable & Defaultable


struct OneToManyAssocView[Assoc: OneToManyAssocLike](Movable):
    var assoc: UnsafePointer[Self.Assoc, MutAnyOrigin]
    var offStorage: UnsafePointer[UInt32, MutAnyOrigin]
    var contentStorage: UnsafePointer[NoneType, MutAnyOrigin]
    var offSize: Int32
    var contentSize: Int32

    fn __init__(out self):
        self.assoc = UnsafePointer[Self.Assoc, MutAnyOrigin]()
        self.offStorage = UnsafePointer[UInt32, MutAnyOrigin]()
        self.contentStorage = UnsafePointer[NoneType, MutAnyOrigin]()
        self.offSize = -1
        self.contentSize = -1

    fn __moveinit__(out self, deinit take: Self):
        self.assoc = take.assoc
        self.offStorage = take.offStorage
        self.contentStorage = take.contentStorage
        self.offSize = take.offSize
        self.contentSize = take.contentSize

    # Type-checked write into the otherwise-opaque contentStorage: callers
    # must supply exactly Self.Assoc.index_type, enforced by the compiler at
    # the call site (this is safe here because View isn't self-referential).
    fn setContentStorage(mut self, ptr: UnsafePointer[Self.Assoc.index_type, MutAnyOrigin]):
        self.contentStorage = ptr.bitcast[NoneType]()


struct OneToManyAssoc[
    I: Copyable & Movable & ImplicitlyCopyable & Defaultable, ONES: Int, SIZE: Int
](Movable, OneToManyAssocLike):
    comptime View = OneToManyAssocView[Self]
    comptime Counter = UInt32
    comptime CountersOnly = OneToManyAssoc[Self.I, Self.ONES, 0]
    comptime index_type = Self.I

    var off: FlexiStorage[Self.Counter, Self.ONES]
    var psws: Int32
    var content: FlexiStorage[Self.index_type, Self.SIZE]

    fn __init__(out self):
        self.off = FlexiStorage[Self.Counter, Self.ONES]()
        self.psws = 0
        self.content = FlexiStorage[Self.index_type, Self.SIZE]()

    fn __moveinit__(out self, deinit take: Self):
        self.off = take.off^
        self.psws = take.psws
        self.content = take.content^

    @staticmethod
    fn ilog2(x_in: UInt32) -> UInt32:
        var v = x_in
        comptime b: InlineArray[UInt32, 5] = [0x2, 0xC, 0xF0, 0xFF00, 0xFFFF0000]
        comptime s: InlineArray[UInt32, 5] = [1, 2, 4, 8, 16]
        var r: UInt32 = 0
        for idx in range(4, -1, -1):
            if v & b[idx]:
                v >>= s[idx]
                r |= s[idx]
        return r

    @staticmethod
    fn ctNOnes() -> Int:
        return Self.ONES

    fn totOnes(self) -> Int:
        return self.off.capacity()

    fn nOnes(self) -> Int:
        return self.totOnes() - 1

    @staticmethod
    fn ctCapacity() -> Int:
        return Self.SIZE

    # C++: void initStorage(View view) -- OneToManyAssoc.h:171-183
    fn initStorage(mut self, view: Self.View):
        debug_assert(
            view.assoc == UnsafePointer(to=self), "OneToManyAssoc.initStorage: view.assoc != this"
        )
        if Self.ctCapacity() < 0:
            debug_assert(
                view.contentStorage != UnsafePointer[NoneType, MutAnyOrigin](),
                "OneToManyAssoc.initStorage: contentStorage is null",
            )
            debug_assert(view.contentSize > 0, "OneToManyAssoc.initStorage: contentSize must be > 0")
            self.content.init(view.contentStorage.bitcast[Self.index_type](), Int(view.contentSize))
        if Self.ctNOnes() < 0:
            debug_assert(
                view.offStorage != UnsafePointer[Self.Counter, MutAnyOrigin](),
                "OneToManyAssoc.initStorage: offStorage is null",
            )
            debug_assert(view.offSize > 0, "OneToManyAssoc.initStorage: offSize must be > 0")
            self.off.init(view.offStorage, Int(view.offSize))

    fn add(mut self, read co: Self.CountersOnly):
        for i in range(self.totOnes()):
            _ = atomic_fetch_add(self.off.data() + i, co.off[i])

    fn zero(mut self):
        for i in range(self.totOnes()):
            self.off[i] = 0

    @staticmethod
    fn atomicIncrement(ptr: UnsafePointer[UInt32, MutAnyOrigin]) -> UInt32:
        return atomic_fetch_add(ptr, UInt32(1))

    @staticmethod
    fn atomicDecrement(ptr: UnsafePointer[UInt32, MutAnyOrigin]) -> UInt32:
        return atomic_fetch_sub(ptr, UInt32(1))

    fn count(mut self, b: Int):
        debug_assert(b < self.nOnes(), "OneToManyAssoc.count: bin out of range")
        _ = Self.atomicIncrement(self.off.data() + b)

    fn fill(mut self, b: Int, j: Self.index_type):
        debug_assert(b < self.nOnes(), "OneToManyAssoc.fill: bin out of range")
        var w = Self.atomicDecrement(self.off.data() + b)
        debug_assert(w > 0, "OneToManyAssoc.fill: bin already empty")
        self.content[Int(w) - 1] = j

    fn bulkFill(
        mut self,
        mut apc: AtomicPairCounter,
        v: UnsafePointer[Self.index_type, MutAnyOrigin],
        n: UInt32,
    ) -> Int32:
        var c = apc.add(n)
        if Int(c[1]) >= self.nOnes():
            return -Int32(c[1])
        self.off[Int(c[1])] = c[0]
        for j in range(n):
            self.content[Int(c[0]) + Int(j)] = v[j]
        return Int32(c[1])

    fn bulkFinalize(mut self, read apc: AtomicPairCounter):
        var c = apc.get()
        self.off[Int(c[1])] = c[0]

    fn bulkFinalizeFill(mut self, read apc: AtomicPairCounter):
        from std.gpu import thread_idx, block_idx, block_dim, grid_dim

        var c = apc.get()
        var m = Int(c[1])
        var n = c[0]
        if m >= self.nOnes():
            self.off[self.nOnes()] = self.off[self.nOnes() - 1]
            return
        var i = m + Int(block_dim.x) * Int(block_idx.x) + Int(thread_idx.x)
        while i < self.totOnes():
            self.off[i] = n
            i += Int(grid_dim.x) * Int(block_dim.x)

    # C++: finalize(Counter *ws = nullptr) -- OneToManyAssoc.h:259-263
    fn finalize[
        ws_address_space: AddressSpace
    ](mut self, ws: UnsafePointer[Self.Counter, MutAnyOrigin, address_space=ws_address_space]):
        debug_assert(self.off[self.totOnes() - 1] == 0, "OneToManyAssoc.finalize: last off must be 0")
        blockPrefixScan(self.off.data(), UInt32(self.totOnes()), ws)
        debug_assert(
            self.off[self.totOnes() - 1] == self.off[self.totOnes() - 2],
            "OneToManyAssoc.finalize: scan did not converge",
        )

    fn size(mut self) -> UInt32:
        return self.off[self.totOnes() - 1]

    fn size(mut self, b: Int) -> UInt32:
        return self.off[b + 1] - self.off[b]

    fn begin(mut self) -> UnsafePointer[Self.index_type, MutAnyOrigin]:
        return self.content.data()

    fn end(mut self) -> UnsafePointer[Self.index_type, MutAnyOrigin]:
        return self.begin() + Int(self.size())

    fn begin(mut self, b: Int) -> UnsafePointer[Self.index_type, MutAnyOrigin]:
        return self.content.data() + Int(self.off[b])

    fn end(mut self, b: Int) -> UnsafePointer[Self.index_type, MutAnyOrigin]:
        return self.content.data() + Int(self.off[b + 1])
