# Mojo port of plugin-Validation/SimpleAtomicHisto.h.
from os.atomic import Atomic, Consistency


# Stands in for C++'s std::atomic<int>: Atomic itself is not movable, so it
# cannot go in a List directly; this wrapper restores that by loading the
# value out and reconstructing on copy/move.
struct AtomicInt(Copyable, Defaultable, ImplicitlyCopyable, Movable):
    var _v: Atomic[DType.int32]

    fn __init__(out self):
        self._v = Atomic[DType.int32](0)

    fn __init__(out self, value: Int32):
        self._v = Atomic[DType.int32](value)

    fn __copyinit__(out self, copy: Self):
        self._v = Atomic[DType.int32](copy._v.load())

    fn __moveinit__(out self, deinit take: Self):
        self._v = Atomic[DType.int32](take._v.load())

    fn load(self) -> Int32:
        return self._v.load()

    fn __iadd__(mut self, value: Int32):
        _ = self._v.fetch_add[ordering = Consistency.SEQUENTIAL](value)


struct SimpleAtomicHisto(Copyable, Defaultable, ImplicitlyCopyable, Movable):
    var data_: List[AtomicInt]
    var min_: Float32
    var max_: Float32

    fn __init__(out self):
        self.data_ = List[AtomicInt]()
        self.min_ = 0.0
        self.max_ = 0.0

    fn __init__(out self, nbins: Int, min: Float32, max: Float32):
        self.data_ = List[AtomicInt](length=nbins + 2, fill=AtomicInt(0))
        self.min_ = min
        self.max_ = max

    # dirty -- C++ rebuilds an empty buffer of the same size here instead of
    # carrying the counts over. Mojo elides the call when the source is dead,
    # so unlike C++ the data survives a last-use move.
    fn __copyinit__(out self, copy: Self):
        self.data_ = List[AtomicInt](length=len(copy.data_), fill=AtomicInt(0))
        self.min_ = copy.min_
        self.max_ = copy.max_

    fn __moveinit__(out self, deinit take: Self):
        self.data_ = List[AtomicInt](length=len(take.data_), fill=AtomicInt(0))
        self.min_ = take.min_
        self.max_ = take.max_

    # thread safe
    fn fill(mut self, value: Float32) raises:
        var size = len(self.data_)
        var i: Int
        if value < self.min_:
            i = 0
        elif value >= self.max_:
            i = size - 1
        else:
            i = Int(
                (value - self.min_)
                / (self.max_ - self.min_)
                * Float32(size - 2)
            )
            # handle rounding near maximum
            if i == size - 2:
                i = size - 3
            if not (i >= 0 and i < size - 2):
                raise Error(
                    "SimpleAtomicHisto::fill(",
                    value,
                    "): i ",
                    i,
                    " min ",
                    self.min_,
                    " max ",
                    self.max_,
                    " nbins ",
                    size - 2,
                )
            i += 1

        debug_assert(i >= 0 and i < size, "SimpleAtomicHisto.fill: i in range")
        self.data_[i] += 1

    fn dump(self) -> String:
        var os = String(len(self.data_)) + " " + String(self.min_) + " " + String(self.max_)
        for item in self.data_:
            os += " " + String(item.load())
        return os

    fn __str__(self) -> String:
        return self.dump()