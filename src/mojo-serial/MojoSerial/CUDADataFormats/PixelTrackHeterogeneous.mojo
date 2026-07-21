from MojoSerial.CUDACore.HistoContainer import OneToManyAssoc
from MojoSerial.CUDACore.EigenSoA import ScalarSoA
from MojoSerial.CUDADataFormats.HeterogeneousSoA import HeterogeneousSoA
from MojoSerial.CUDADataFormats.TrajectoryStateSoA import TrajectoryStateSoA
from MojoSerial.MojoBridge.Matrix import Vector, Matrix
from MojoSerial.MojoBridge.DTypes import Float, Typeable


@nonmaterializable(NoneType)
struct TrackQuality:
    alias bad: UInt8 = 0
    alias dup: UInt8 = 1
    alias loose: UInt8 = 2
    alias strict: UInt8 = 3
    alias tight: UInt8 = 4
    alias highPurity: UInt8 = 5


# WARNING: THIS STRUCT IS 128-ALIGNED (ScalarSoA)
@fieldwise_init
struct TrackSoAT[S: Int32](Defaultable, Movable, Typeable):
    @staticmethod
    @always_inline
    fn stride() -> Int32:
        return S

    alias Quality = UInt8
    alias HIndexType = DType.uint16
    alias HitContainer = OneToManyAssoc[
        Self.HIndexType,
        S.cast[DType.uint32](),
        5 * S.cast[DType.uint32](),
    ]

    var m_quality: ScalarSoA[DType.uint8, Int(S)]

    # this is chi2/ndof as not necessarely all hits are used in the fit
    var chi2: ScalarSoA[DType.float32, Int(S)]

    # State at the Beam spot
    # phi,tip,1/pt,cotan(theta),zip
    var stateAtBS: TrajectoryStateSoA[S]
    var eta: ScalarSoA[DType.float32, Int(S)]
    var pt: ScalarSoA[DType.float32, Int(S)]

    # state at the detector of the outermost hit
    var hitIndices: Self.HitContainer
    var detIndices: Self.HitContainer

    # total number of tracks (including those not fitted)
    var m_nTracks: UInt32

    @always_inline
    fn __init__(out self):
        self.m_quality = ScalarSoA[DType.uint8, Int(S)]()

        self.chi2 = ScalarSoA[DType.float32, Int(S)]()

        self.stateAtBS = TrajectoryStateSoA[S]()
        self.eta = ScalarSoA[DType.float32, Int(S)]()
        self.pt = ScalarSoA[DType.float32, Int(S)]()

        self.hitIndices = Self.HitContainer()
        self.detIndices = Self.HitContainer()
        self.m_nTracks = 0

    @always_inline
    fn quality(ref self, i: Int) -> ref [self.m_quality._data] Self.Quality:
        return self.m_quality[i]

    @always_inline
    fn qualityData[
        origin: Origin, //
    ](ref [origin]self) -> UnsafePointer[
        Self.Quality, mut = origin.mut, origin=origin
    ]:
        return self.m_quality.data()

    @always_inline
    fn nHits(self, i: Int32) -> Int32:
        return self.detIndices.size(i.cast[DType.uint32]()).cast[DType.int32]()

    @always_inline
    fn charge(self, i: Int32) -> Float:
        return Float(1.0) if self.stateAtBS.state[i][2, 0] >= 0 else Float(-1.0)

    @always_inline
    fn phi(self, i: Int32) -> Float:
        return rebind[Scalar[DType.float32]](self.stateAtBS.state[i][0, 0])

    @always_inline
    fn tip(self, i: Int32) -> Float:
        return rebind[Scalar[DType.float32]](self.stateAtBS.state[i][1, 0])

    @always_inline
    fn zip(self, i: Int32) -> Float:
        return rebind[Scalar[DType.float32]](self.stateAtBS.state[i][4, 0])

    @always_inline
    @staticmethod
    fn dtype() -> String:
        return "TrackSoAT[" + String(S) + "]"


@nonmaterializable(NoneType)
struct PixelTrack:
    @staticmethod
    @always_inline
    fn maxNumber() -> UInt32:
        return 32 * 1024

    alias TrackSoA = TrackSoAT[Self.maxNumber().cast[DType.int32]()]
    alias TrajectoryState = TrajectoryStateSoA[
        Self.maxNumber().cast[DType.int32]()
    ]
    alias HitContainer = Self.TrackSoA.HitContainer
    alias Quality = UInt8


alias PixelTrackHeterogeneous = HeterogeneousSoA[PixelTrack.TrackSoA]
