from CUDADataFormats import PixelTrackHeterogeneous
import CAConstants
import FitResults
from MojoSerial.CUDACore.CUDACompat import CUDAStreamType
from MojoSerial.MojoBridge.Matrix import Matrix, Map
from MojoSerial.CUDADataFormats.TrackingRecHit2DSOAView import TrackingRecHit2DSOAView
from MojoSerial.CUDADataFormats.PixelTrackHeterogeneous import PixelTrack as pixelTrack

alias cudaStream_t = CUDAStreamType

struct Rfit:
    # in case of memory issue can be made smaller
    @staticmethod
    @parameter
    fn maxNumberOfConcurrentFits() -> UInt32:
        return CAConstants.maxNumberOfTuples()

    @staticmethod
    @parameter
    fn stride() -> UInt32:
        return Rfit.maxNumberOfConcurrentFits()

    alias Matrix3x4d = Matrix[DType.float64, 3, 4]
    alias Map3x4d = Map[
        Float64,
        3,
        4,
        stride(),
    ]
    alias Matrix6x4f = Matrix[DType.float32, 6, 4]
    alias Map6x4f = Map[
        Float32,
        6,
        4,
        stride(),
    ]

    # hits
    alias Matrix3xNd[N: Int] = Matrix[DType.float64, 3, N]
    alias Map3xNd[N: Int] = Map[
        Float64,
        3,
        N,
        stride(),
    ]

    # errors
    alias Matrix6xNf[N: Int] = Matrix[DType.float32, 6, N]
    alias Map6xNf[N: Int] = Map[
        Float32,
        6,
        N,
        stride(),
    ]

    # fast fit
    alias Vector4d = Matrix[DType.float64, 4, 1]
    alias Map4d = Map[Float64, 4, 1, stride()]


trait HelixFitOnGPU:
    alias HitsView = TrackingRecHit2DSOAView

    alias Tuples = pixelTrack.HitContainer
    alias OutputSoA = pixelTrack.TrackSoA

    alias TupleMultiplicity = CAConstants.TupleMultiplicity

    fn setBField(mut self, bField: Float64):
        self.bField_ = Float32(bField)

    fn launchRiemannKernels(
        self,
        hv: UnsafePointer[HitsView],
        nhits: UInt32,
        maxNumberOfTuples: UInt32,
        cudaStream: cudaStream_t
    ): ...

    fn launchBrokenLineKernels(
        self,
        hv: UnsafePointer[HitsView],
        nhits: UInt32,
        maxNumberOfTuples: UInt32,
        cudaStream: cudaStream_t
    ): ...

    fn launchRiemannKernelsOnCPU(
        self,
        hv: UnsafePointer[HitsView],
        nhits: UInt32,
        maxNumberOfTuples: UInt32
    ): ...

    fn launchBrokenLineKernelsOnCPU(
        self,
        hv: UnsafePointer[HitsView],
        nhits: UInt32,
        maxNumberOfTuples: UInt32
    ): ...

    fn allocateOnGPU(
        mut self,
        tuples: UnsafePointer[Tuples],
        tupleMultiplicity: UnsafePointer[TupleMultiplicity],
        outputSoA: UnsafePointer[OutputSoA]
    ): ...

    fn deallocateOnGPU(mut self): ...

    alias maxNumberOfConcurrentFits_ = Rfit.maxNumberOfConcurrentFits()

    var tuples_d: UnsafePointer[Tuples]
    var tupleMultiplicity_d: UnsafePointer[TupleMultiplicity]
    var outputSoa_d: UnsafePointer[OutputSoA]

    var bField_: Float32
    var fit5as4_: Bool
