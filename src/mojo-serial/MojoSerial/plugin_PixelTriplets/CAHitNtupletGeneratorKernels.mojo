import CAConstants
from GPUCACell import GPUCACell
from MojoSerial.CUDACore.AtomicPairCounter import AtomicPairCounter
from MojoSerial.CUDACore.CUDACompat import CUDAStreamType
from MojoSerial.CUDADataFormats.PixelTrackHeterogeneous import (
    PixelTrack as pixelTrack,
)
from MojoSerial.CUDADataFormats.TrackingRecHit2DHeterogeneous import (
    TrackingRecHit2DHeterogeneous,
)
from MojoSerial.CUDADataFormats.TrackingRecHit2DSOAView import (
    TrackingRecHit2DSOAView,
)

alias HitToTuple = CAConstants.HitToTuple
alias TupleMultiplicity = CAConstants.TupleMultiplicity
alias Quality = pixelTrack.Quality
alias TkSoA = pixelTrack.TrackSoA
alias HitContainer = pixelTrack.HitContainer
alias Stream = CUDAStreamType

struct Counters:
    var nEvents: UInt64
    var nHits: UInt64
    var nCells: UInt64
    var nTuples: UInt64
    var nFitTracks: UInt64
    var nGoodTracks: UInt64
    var nUsedHits: UInt64
    var nDupHits: UInt64
    var nKilledCells: UInt64
    var nEmptyCells: UInt64
    var nZeroTrackCells: UInt64

struct QualityCuts:
    # chi2 cut = chi2Scale * (chi2Coeff[0] + pT/GeV * (chi2Coeff[1] + pT/GeV * (chi2Coeff[2] + pT/GeV * chi2Coeff[3])))
    var chi2Coeff: Float32[4]
    var chi2MaxPt: Float32  # GeV
    var chi2Scale: Float32

    struct region:
        var maxTip: Float32  # cm
        var minPt: Float32   # GeV
        var maxZip: Float32  # cm

    var triplet: region
    var quadruplet: region


struct Params:
    var onGPU: Bool
    var minHitsPerNtuplet: UInt32
    var maxNumberOfDoublets: UInt32
    var useRiemannFit: Bool
    var fit5as4: Bool
    var includeJumpingForwardDoublets: Bool
    var earlyFishbone: Bool
    var lateFishbone: Bool
    var idealConditions: Bool
    var doStats: Bool
    var doClusterCut: Bool
    var doZ0Cut: Bool
    var doPtCut: Bool
    var ptmin: Float32
    var CAThetaCutBarrel: Float32
    var CAThetaCutForward: Float32
    var hardCurvCut: Float32
    var dcaCutInnerTriplet: Float32
    var dcaCutOuterTriplet: Float32
    var cuts: QualityCuts

    fn __init__(
        onGPU: Bool,
        minHitsPerNtuplet: UInt32,
        maxNumberOfDoublets: UInt32,
        useRiemannFit: Bool,
        fit5as4: Bool,
        includeJumpingForwardDoublets: Bool,
        earlyFishbone: Bool,
        lateFishbone: Bool,
        idealConditions: Bool,
        doStats: Bool,
        doClusterCut: Bool,
        doZ0Cut: Bool,
        doPtCut: Bool,
        ptmin: Float32,
        CAThetaCutBarrel: Float32,
        CAThetaCutForward: Float32,
        hardCurvCut: Float32,
        dcaCutInnerTriplet: Float32,
        dcaCutOuterTriplet: Float32,
        cuts: QualityCuts = QualityCuts(
            #polynomial coefficients for the pT-dependent chi2 cut
            chi2Coeff = [0.68177776, 0.74609577, -0.08035491, 0.00315399],
            #max pT used to determine the chi2 cut
            chi2MaxPt = 10.0,
            #chi2 scale factor: 30 for broken line fit, 45 for Riemann fit
            chi2Scale = 30.0,
            #regional cuts for triplets
            triplet = QualityCuts.region(maxTip = 0.3, minPt = 0.5, maxZip = 12.0),
           #regional cuts for quadruplets
            quadruplet = QualityCuts.region(maxTip = 0.5, minPt = 0.3, maxZip = 12.0)
        )
    ):
        self.onGPU = onGPU
        self.minHitsPerNtuplet = minHitsPerNtuplet
        self.maxNumberOfDoublets = maxNumberOfDoublets
        self.useRiemannFit = useRiemannFit
        self.fit5as4 = fit5as4
        self.includeJumpingForwardDoublets = includeJumpingForwardDoublets
        self.earlyFishbone = earlyFishbone
        self.lateFishbone = lateFishbone
        self.idealConditions = idealConditions
        self.doStats = doStats
        self.doClusterCut = doClusterCut
        self.doZ0Cut = doZ0Cut
        self.doPtCut = doPtCut
        self.ptmin = ptmin
        self.CAThetaCutBarrel = CAThetaCutBarrel
        self.CAThetaCutForward = CAThetaCutForward
        self.hardCurvCut = hardCurvCut
        self.dcaCutInnerTriplet = dcaCutInnerTriplet
        self.dcaCutOuterTriplet = dcaCutOuterTriplet
        self.cuts = cuts

trait MemoryTraits:
    fn unique_ptr[T]() -> T*


# --- Main Struct (GPU Kernel Driver) ---
trait CAHitNtupletGeneratorKernels[T: MemoryTraits]:

    # Type aliases
    alias HitsView = TrackingRecHit2DSOAView
    alias HitsOnGPU = TrackingRecHit2DSOAView
    #mojo version of TrackingRecHit2DHeterogeneous is not generic 
    alias HitsOnCPU = TrackingRecHit2DHeterogeneous
    alias HitToTuple = HitToTuple
    alias TupleMultiplicity = TupleMultiplicity
    alias Quality = Quality
    alias TkSoA = TrackSoA
    alias HitContainer = HitContainer
    alias unique_ptr[X] = T.unique_ptr[X]



    var counters_: Counters* = None

    # --- Workspace  ---
    var cellStorage_: unique_ptr[UInt8[]]
    var device_theCellNeighbors_: unique_ptr[CAConstants.CellNeighborsVector]
    var device_theCellNeighborsContainer_: Nullable[CAConstants.CellNeighbors*]
    var device_theCellTracks_: unique_ptr[CAConstants.CellTracksVector]
    var device_theCellTracksContainer_: Nullable[CAConstants.CellTracks*]

    var device_theCells_: unique_ptr[GPUCACell[]]
    var device_isOuterHitOfCell_: unique_ptr[GPUCACell.OuterHitOfCell[]]
    var device_nCells_: Nullable[UnsafePointer[UInt32]] = None

    var device_hitToTuple_: unique_ptr[HitToTuple]
    var device_hitToTuple_apc_: Nullable[AtomicPairCounter*] = None

    var device_hitTuple_apc_: Nullable[AtomicPairCounter*] = None

    var device_tupleMultiplicity_: unique_ptr[TupleMultiplicity]
    var device_storage_: unique_ptr[AtomicPairCounter.c_type[]]


    var m_params: borrowed[Params]


    # --- Constructor ---
    fn __init__(params: Params):
        self.m_params = params

    # --- Methods (signatures only for now) ---
    fn tuple_multiplicity(self) -> TupleMultiplicity*:
        return self.device_tupleMultiplicity_

    fn launch_kernels(
        self,
        hh: HitsOnCPU,
        tuples_d: TrackSoA*,
        cudaSream: Stream
    ) raises:
        pass

    fn classify_tuples(
        self,
        hh: HitsOnCPU,
        tuples_d: TrackSoA*,
        cudaSream: Stream
    ) raises:
        pass

    fn fill_hit_det_indices(
        self,
        hv: HitsView*,
        tuples_d: TrackSoA*,
        cudaSream: Stream
    ) raises:
        pass

    fn build_doublets(
        self,
        hh: HitsOnCPU,
        stream: Stream
    ) raises:
        pass

    fn allocate_on_gpu(self, stream: Stream) raises:
        pass

    fn cleanup(self, cudaSream: Stream) raises:
        pass

    @staticmethod
    fn print_counters(counters: Counters*):
        pass

comptime CAHitNtupletGeneratorKernelsCPU= CAHitNtupletGeneratorKernels[CPUTraits]
comptime CAHitNtupletGeneratorKernelsGPU= CAHitNtupletGeneratorKernels[GPUTraits]
