from MojoSerial.CUDADataFormats.PixelTrackHeterogeneous import PixelTrackHeterogeneous
import MojoSerial.plugin_PixelTriplets.CAConstants as CAConstants
import MojoSerial.plugin_PixelTriplets.FitResults as FitResults
from MojoSerial.CUDACore.CUDACompat import CUDAStreamType
from MojoSerial.MojoBridge.Matrix import Matrix, Map
from MojoSerial.CUDADataFormats.TrackingRecHit2DSOAView import TrackingRecHit2DSOAView
from MojoSerial.CUDADataFormats.PixelTrackHeterogeneous import PixelTrack as pixelTrack
from MojoSerial.plugin_PixelTriplets.FitUtils import Rfit as FitUtilsRfit
from MojoSerial.plugin_PixelTriplets.RiemannFitOnGPU import (
    kernelFastFit,
    kernelCircleFit,
    kernelLineFit,
)
from MojoSerial.plugin_PixelTriplets.BrokenLineFitOnGPU import (
    kernelBLFastFit,
    kernelBLFit,
)

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
        DType.float64,
        3,
        4,
        Int(Self.stride()),
    ]
    alias Matrix6x4f = Matrix[DType.float32, 6, 4]
    alias Map6x4f = Map[
        DType.float32,
        6,
        4,
        Int(Self.stride()),
    ]

    # hits
    alias Matrix3xNd[N: Int] = Matrix[DType.float64, 3, N]
    alias Map3xNd[N: Int] = Map[
        DType.float64,
        3,
        N,
        Int(Self.stride()),
    ]

    # errors
    alias Matrix6xNf[N: Int] = Matrix[DType.float32, 6, N]
    alias Map6xNf[N: Int] = Map[
        DType.float32,
        6,
        N,
        Int(Self.stride()),
    ]

    # fast fit
    alias Vector4d = Matrix[DType.float64, 4, 1]
    alias Map4d = Map[DType.float64, 4, 1, Int(Self.stride())]

    alias Matrix3d = Matrix[DType.float64, 3, 3]
    alias line_fit = FitUtilsRfit.line_fit


# C++: HelixFitOnGPU is a plain (non-template) class -- ported directly as a
# struct with real methods, same fix as CAHitNtupletGeneratorKernelsCPU
# (a `trait` with `var` fields is invalid Mojo: "fields in traits are not
# supported yet").
struct HelixFitOnGPU:
    alias HitsView = TrackingRecHit2DSOAView

    alias Tuples = pixelTrack.HitContainer
    alias OutputSoA = pixelTrack.TrackSoA

    alias TupleMultiplicity = CAConstants.TupleMultiplicity

    alias maxNumberOfConcurrentFits_ = Rfit.maxNumberOfConcurrentFits()

    var tuples_d: UnsafePointer[Self.Tuples]
    var tupleMultiplicity_d: UnsafePointer[Self.TupleMultiplicity]
    var outputSoa_d: UnsafePointer[Self.OutputSoA]

    var bField_: Float32
    var fit5as4_: Bool

    fn __init__(out self, bf: Float32, fit5as4: Bool):
        self.bField_ = bf
        self.fit5as4_ = fit5as4
        self.tuples_d = UnsafePointer[Self.Tuples]()
        self.tupleMultiplicity_d = UnsafePointer[Self.TupleMultiplicity]()
        self.outputSoa_d = UnsafePointer[Self.OutputSoA]()

    fn setBField(mut self, bField: Float64):
        self.bField_ = Float32(bField)

    # C++ only declares these (GPU-kernel-launch wrappers); no .cc anywhere in
    # the serial backend ever defines them -- the CPU backend always calls
    # the *OnCPU variants below instead (see CAHitNtupletGeneratorOnGPU.cc).
    fn launchRiemannKernels(
        self,
        hv: UnsafePointer[Self.HitsView],
        nhits: UInt32,
        maxNumberOfTuples: UInt32,
        cudaStream: cudaStream_t
    ):
        pass

    fn launchBrokenLineKernels(
        self,
        hv: UnsafePointer[Self.HitsView],
        nhits: UInt32,
        maxNumberOfTuples: UInt32,
        cudaStream: cudaStream_t
    ):
        pass

    # C++: HelixFitOnGPU::launchRiemannKernelsOnCPU (RiemannFitOnGPU.cc)
    fn launchRiemannKernelsOnCPU(
        self,
        hv: UnsafePointer[Self.HitsView],
        nhits: UInt32,
        maxNumberOfTuples: UInt32
    ) raises:
        debug_assert(Bool(self.tuples_d))

        var nConcurrent = Int(Self.maxNumberOfConcurrentFits_)
        # Fit internals -- C++ sizes these as
        # maxNumberOfConcurrentFits_ * sizeof(Rfit::Matrix3xNd<4>) / sizeof(double)
        # etc.; ported directly as element counts (List[T] is this port's
        # unique_ptr<T[]> equivalent) instead of raw byte arithmetic.
        var hitsGPU_ = List[Float64](length=nConcurrent * 3 * 4, fill=0.0)
        var hits_geGPU_ = List[Float32](length=nConcurrent * 6 * 4, fill=0.0)
        var fast_fit_resultsGPU_ = List[Float64](length=nConcurrent * 4, fill=0.0)
        var circle_fit_resultsGPU_ = List[FitUtilsRfit.circle_fit](
            length=nConcurrent, fill=FitUtilsRfit.circle_fit()
        )

        var offset: UInt32 = 0
        while offset < maxNumberOfTuples:
            # triplets
            kernelFastFit[3](
                self.tuples_d, self.tupleMultiplicity_d, 3, hv,
                hitsGPU_.unsafe_ptr(), hits_geGPU_.unsafe_ptr(),
                fast_fit_resultsGPU_.unsafe_ptr(), offset,
            )
            kernelCircleFit[3](
                self.tupleMultiplicity_d, 3, Float64(self.bField_),
                hitsGPU_.unsafe_ptr(), hits_geGPU_.unsafe_ptr(),
                fast_fit_resultsGPU_.unsafe_ptr(),
                circle_fit_resultsGPU_.unsafe_ptr(), offset,
            )
            kernelLineFit[3](
                self.tupleMultiplicity_d, 3, Float64(self.bField_),
                self.outputSoa_d, hitsGPU_.unsafe_ptr(),
                hits_geGPU_.unsafe_ptr(), fast_fit_resultsGPU_.unsafe_ptr(),
                circle_fit_resultsGPU_.unsafe_ptr(), offset,
            )

            # quads
            kernelFastFit[4](
                self.tuples_d, self.tupleMultiplicity_d, 4, hv,
                hitsGPU_.unsafe_ptr(), hits_geGPU_.unsafe_ptr(),
                fast_fit_resultsGPU_.unsafe_ptr(), offset,
            )
            kernelCircleFit[4](
                self.tupleMultiplicity_d, 4, Float64(self.bField_),
                hitsGPU_.unsafe_ptr(), hits_geGPU_.unsafe_ptr(),
                fast_fit_resultsGPU_.unsafe_ptr(),
                circle_fit_resultsGPU_.unsafe_ptr(), offset,
            )
            kernelLineFit[4](
                self.tupleMultiplicity_d, 4, Float64(self.bField_),
                self.outputSoa_d, hitsGPU_.unsafe_ptr(),
                hits_geGPU_.unsafe_ptr(), fast_fit_resultsGPU_.unsafe_ptr(),
                circle_fit_resultsGPU_.unsafe_ptr(), offset,
            )

            if self.fit5as4_:
                # penta (only first 4)
                kernelFastFit[4](
                    self.tuples_d, self.tupleMultiplicity_d, 5, hv,
                    hitsGPU_.unsafe_ptr(), hits_geGPU_.unsafe_ptr(),
                    fast_fit_resultsGPU_.unsafe_ptr(), offset,
                )
                kernelCircleFit[4](
                    self.tupleMultiplicity_d, 5, Float64(self.bField_),
                    hitsGPU_.unsafe_ptr(), hits_geGPU_.unsafe_ptr(),
                    fast_fit_resultsGPU_.unsafe_ptr(),
                    circle_fit_resultsGPU_.unsafe_ptr(), offset,
                )
                kernelLineFit[4](
                    self.tupleMultiplicity_d, 5, Float64(self.bField_),
                    self.outputSoa_d, hitsGPU_.unsafe_ptr(),
                    hits_geGPU_.unsafe_ptr(),
                    fast_fit_resultsGPU_.unsafe_ptr(),
                    circle_fit_resultsGPU_.unsafe_ptr(), offset,
                )
            else:
                # penta (all 5)
                kernelFastFit[5](
                    self.tuples_d, self.tupleMultiplicity_d, 5, hv,
                    hitsGPU_.unsafe_ptr(), hits_geGPU_.unsafe_ptr(),
                    fast_fit_resultsGPU_.unsafe_ptr(), offset,
                )
                kernelCircleFit[5](
                    self.tupleMultiplicity_d, 5, Float64(self.bField_),
                    hitsGPU_.unsafe_ptr(), hits_geGPU_.unsafe_ptr(),
                    fast_fit_resultsGPU_.unsafe_ptr(),
                    circle_fit_resultsGPU_.unsafe_ptr(), offset,
                )
                kernelLineFit[5](
                    self.tupleMultiplicity_d, 5, Float64(self.bField_),
                    self.outputSoa_d, hitsGPU_.unsafe_ptr(),
                    hits_geGPU_.unsafe_ptr(),
                    fast_fit_resultsGPU_.unsafe_ptr(),
                    circle_fit_resultsGPU_.unsafe_ptr(), offset,
                )

            offset += Self.maxNumberOfConcurrentFits_

    # C++: HelixFitOnGPU::launchBrokenLineKernelsOnCPU (BrokenLineFitOnGPU.cc)
    fn launchBrokenLineKernelsOnCPU(
        self,
        hv: UnsafePointer[Self.HitsView],
        nhits: UInt32,
        maxNumberOfTuples: UInt32
    ) raises:
        debug_assert(Bool(self.tuples_d))

        var nConcurrent = Int(Self.maxNumberOfConcurrentFits_)
        var hitsGPU_ = List[Float64](length=nConcurrent * 3 * 4, fill=0.0)
        var hits_geGPU_ = List[Float32](length=nConcurrent * 6 * 4, fill=0.0)
        var fast_fit_resultsGPU_ = List[Float64](length=nConcurrent * 4, fill=0.0)

        var offset: UInt32 = 0
        while offset < maxNumberOfTuples:
            # fit triplets
            kernelBLFastFit[3](
                self.tuples_d, self.tupleMultiplicity_d, hv,
                hitsGPU_.unsafe_ptr(), hits_geGPU_.unsafe_ptr(),
                fast_fit_resultsGPU_.unsafe_ptr(), 3, offset,
            )
            kernelBLFit[3](
                self.tupleMultiplicity_d, Float64(self.bField_),
                self.outputSoa_d, hitsGPU_.unsafe_ptr(),
                hits_geGPU_.unsafe_ptr(), fast_fit_resultsGPU_.unsafe_ptr(),
                3, offset,
            )

            # fit quads
            kernelBLFastFit[4](
                self.tuples_d, self.tupleMultiplicity_d, hv,
                hitsGPU_.unsafe_ptr(), hits_geGPU_.unsafe_ptr(),
                fast_fit_resultsGPU_.unsafe_ptr(), 4, offset,
            )
            kernelBLFit[4](
                self.tupleMultiplicity_d, Float64(self.bField_),
                self.outputSoa_d, hitsGPU_.unsafe_ptr(),
                hits_geGPU_.unsafe_ptr(), fast_fit_resultsGPU_.unsafe_ptr(),
                4, offset,
            )

            if self.fit5as4_:
                # fit penta (only first 4)
                kernelBLFastFit[4](
                    self.tuples_d, self.tupleMultiplicity_d, hv,
                    hitsGPU_.unsafe_ptr(), hits_geGPU_.unsafe_ptr(),
                    fast_fit_resultsGPU_.unsafe_ptr(), 5, offset,
                )
                kernelBLFit[4](
                    self.tupleMultiplicity_d, Float64(self.bField_),
                    self.outputSoa_d, hitsGPU_.unsafe_ptr(),
                    hits_geGPU_.unsafe_ptr(),
                    fast_fit_resultsGPU_.unsafe_ptr(), 5, offset,
                )
            else:
                # fit penta (all 5)
                kernelBLFastFit[5](
                    self.tuples_d, self.tupleMultiplicity_d, hv,
                    hitsGPU_.unsafe_ptr(), hits_geGPU_.unsafe_ptr(),
                    fast_fit_resultsGPU_.unsafe_ptr(), 5, offset,
                )
                kernelBLFit[5](
                    self.tupleMultiplicity_d, Float64(self.bField_),
                    self.outputSoa_d, hitsGPU_.unsafe_ptr(),
                    hits_geGPU_.unsafe_ptr(),
                    fast_fit_resultsGPU_.unsafe_ptr(), 5, offset,
                )

            offset += Self.maxNumberOfConcurrentFits_

    fn allocateOnGPU(
        mut self,
        tuples: UnsafePointer[Self.Tuples],
        tupleMultiplicity: UnsafePointer[Self.TupleMultiplicity],
        outputSoA: UnsafePointer[Self.OutputSoA]
    ):
        self.tuples_d = tuples
        self.tupleMultiplicity_d = tupleMultiplicity
        self.outputSoa_d = outputSoA
        debug_assert(Bool(self.tuples_d))
        debug_assert(Bool(self.tupleMultiplicity_d))
        debug_assert(Bool(self.outputSoa_d))

    fn deallocateOnGPU(mut self):
        pass
