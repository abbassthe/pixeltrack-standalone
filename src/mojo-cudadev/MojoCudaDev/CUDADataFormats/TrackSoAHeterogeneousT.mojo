# Mojo port of CUDADataFormats/TrackSoAHeterogeneousT.h. Retargeted at the
# new OneToManyAssoc (not mojo-serial's HistoContainer-based one) and fixed
# to match C++'s hindex_type = uint32_t (mojo-serial used uint16 -- a real
# CMS-code delta the port plan flagged, not a typo).
#
# GPU_SMALL_EVENTS is a build-time #ifdef in C++, undefined by default (the
# `#else` branch is what real builds use) -- modeled as a comptime Bool,
# same idiom as is_gpu() standing in for #ifdef __CUDA_ARCH__ elsewhere in
# this port.
comptime GPU_SMALL_EVENTS = False

from math import copysign

from MojoCudaDev.CUDACore.OneToManyAssoc import OneToManyAssoc
from MojoCudaDev.CUDACore.eigenSoA import ScalarSoA
from MojoCudaDev.CUDADataFormats.TrajectoryStateSoA import TrajectoryStateSoA
from MojoCudaDev.MojoBridge.DTypes import Float, Typeable


# C++: enum class Quality : uint8_t { bad = 0, dup, loose, strict, tight, highPurity };
# -- TrackSoAHeterogeneousT.h:9
struct Quality:
    comptime T = UInt8
    comptime bad: Self.T = 0
    comptime dup: Self.T = 1
    comptime loose: Self.T = 2
    comptime strict: Self.T = 3
    comptime tight: Self.T = 4
    comptime highPurity: Self.T = 5


# C++: namespace pixelTrack { ... } -- TrackSoAHeterogeneousT.h:8-10, 56-70
struct pixelTrack:
    comptime Quality = Quality

    @staticmethod
    fn maxNumber() -> Int:
        if GPU_SMALL_EVENTS:
            return 2 * 1024
        else:
            return 32 * 1024

    comptime TrackSoA = TrackSoAHeterogeneousT[Self.maxNumber()]
    comptime TrajectoryState = TrajectoryStateSoA[Self.maxNumber()]
    comptime HitContainer = Self.TrackSoA.HitContainer


struct TrackSoAHeterogeneousT[S: Int](Movable, Typeable):
    comptime Quality = pixelTrack.Quality.T
    comptime hindex_type = UInt32
    comptime HitContainer = OneToManyAssoc[Self.hindex_type, Self.S + 1, 5 * Self.S]

    var quality_: ScalarSoA[DType.uint8, Self.S]
    var chi2: ScalarSoA[DType.float32, Self.S]
    var stateAtBS: TrajectoryStateSoA[Self.S]
    var eta: ScalarSoA[DType.float32, Self.S]
    var pt: ScalarSoA[DType.float32, Self.S]
    var hitIndices: Self.HitContainer
    var detIndices: Self.HitContainer

    fn __init__(out self):
        self.quality_ = ScalarSoA[DType.uint8, Self.S]()
        self.chi2 = ScalarSoA[DType.float32, Self.S]()
        self.stateAtBS = TrajectoryStateSoA[Self.S]()
        self.eta = ScalarSoA[DType.float32, Self.S]()
        self.pt = ScalarSoA[DType.float32, Self.S]()
        self.hitIndices = Self.HitContainer()
        self.detIndices = Self.HitContainer()

    fn __moveinit__(out self, deinit take: Self):
        self.quality_ = take.quality_^
        self.chi2 = take.chi2^
        self.stateAtBS = take.stateAtBS^
        self.eta = take.eta^
        self.pt = take.pt^
        self.hitIndices = take.hitIndices^
        self.detIndices = take.detIndices^

    @staticmethod
    fn stride() -> Int:
        return Self.S

    fn quality(ref self, i: Int) -> ref [self.quality_._data] Self.Quality:
        return self.quality_[i]

    fn qualityData(mut self) -> UnsafePointer[Self.Quality, MutAnyOrigin]:
        return self.quality_.data()

    fn nHits(mut self, i: Int) -> Int:
        return Int(self.detIndices.size(i))

    fn charge(mut self, i: Int32) -> Float:
        return copysign(
            Float(1.0), rebind[Scalar[DType.float32]](self.stateAtBS.state[i][2, 0].cast[DType.float32]())
        )

    fn phi(mut self, i: Int32) -> Float:
        return rebind[Scalar[DType.float32]](self.stateAtBS.state[i][0, 0].cast[DType.float32]())

    fn tip(mut self, i: Int32) -> Float:
        return rebind[Scalar[DType.float32]](self.stateAtBS.state[i][1, 0].cast[DType.float32]())

    fn zip(mut self, i: Int32) -> Float:
        return rebind[Scalar[DType.float32]](self.stateAtBS.state[i][4, 0].cast[DType.float32]())

    @staticmethod
    fn dtype() -> String:
        return "TrackSoAHeterogeneousT[" + String(Self.S) + "]"
