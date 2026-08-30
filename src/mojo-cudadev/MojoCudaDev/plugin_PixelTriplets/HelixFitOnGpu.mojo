# Mojo port of plugin-PixelTriplets/HelixFitOnGPU.h + HelixFitOnGPU.cc.
from MojoCudaDev.CUDACore.CUDACompat import CUDAStreamType
from MojoCudaDev.CUDADataFormats.TrackSoAHeterogeneousT import pixelTrack
from MojoCudaDev.CUDADataFormats.TrackingRecHit2DSOAView import (
    TrackingRecHit2DSOAView,
)
from MojoCudaDev.MojoBridge.Matrix import Matrix, Map
from MojoCudaDev.plugin_PixelTriplets.CAConstants import caConstants
from MojoCudaDev.plugin_PixelTriplets.FitUtils import (
    riemannFit as riemannFitUtils,
)
from MojoCudaDev.plugin_PixelTriplets.RiemannFitOnGPU import (
    kernelFastFit,
    kernelCircleFit,
    kernelLineFit,
)
from MojoCudaDev.plugin_PixelTriplets.BrokenLineFitOnGPU import (
    kernelBLFastFit,
    kernelBLFit,
)

comptime cudaStream_t = CUDAStreamType


struct riemannFit:
    # in case of memory issue can be made smaller
    comptime maxNumberOfConcurrentFits = caConstants.maxNumberOfTuples
    comptime stride = Self.maxNumberOfConcurrentFits

    comptime Matrix3x4d = Matrix[DType.float64, 3, 4]
    comptime Map3x4d = Map[DType.float64, 3, 4, Int(Self.stride)]
    comptime Matrix6x4f = Matrix[DType.float32, 6, 4]
    comptime Map6x4f = Map[DType.float32, 6, 4, Int(Self.stride)]

    # hits
    comptime Matrix3xNd[N: Int] = Matrix[DType.float64, 3, N]
    comptime Map3xNd[N: Int] = Map[DType.float64, 3, N, Int(Self.stride)]

    # errors
    comptime Matrix6xNf[N: Int] = Matrix[DType.float32, 6, N]
    comptime Map6xNf[N: Int] = Map[DType.float32, 6, N, Int(Self.stride)]

    # fast fit
    comptime Map4d = Map[DType.float64, 4, 1, Int(Self.stride)]


struct HelixFitOnGPU:
    comptime HitsView = TrackingRecHit2DSOAView

    comptime Tuples = pixelTrack.HitContainer
    comptime OutputSoA = pixelTrack.TrackSoA

    comptime TupleMultiplicity = caConstants.TupleMultiplicity

    comptime maxNumberOfConcurrentFits_ = riemannFit.maxNumberOfConcurrentFits

    var tuples_: UnsafePointer[Self.Tuples, MutAnyOrigin]
    var tupleMultiplicity_: UnsafePointer[Self.TupleMultiplicity, MutAnyOrigin]
    var outputSoa_: UnsafePointer[Self.OutputSoA, MutAnyOrigin]

    var bField_: Float32
    var fit5as4_: Bool

    fn __init__(out self, bf: Float32, fit5as4: Bool):
        self.bField_ = bf
        self.fit5as4_ = fit5as4
        self.tuples_ = UnsafePointer[Self.Tuples, MutAnyOrigin]()
        self.tupleMultiplicity_ = UnsafePointer[Self.TupleMultiplicity, MutAnyOrigin]()
        self.outputSoa_ = UnsafePointer[Self.OutputSoA, MutAnyOrigin]()

    fn setBField(mut self, bField: Float64):
        self.bField_ = Float32(bField)

    # Defined in cudadev by RiemannFitOnGPU.cu / BrokenLineFitOnGPU.cu, which
    # are not ported yet.
    fn launchRiemannKernels(
        self,
        hv: UnsafePointer[Self.HitsView, MutAnyOrigin],
        nhits: UInt32,
        maxNumberOfTuples: UInt32,
        cudaStream: cudaStream_t,
    ):
        pass

    fn launchBrokenLineKernels(
        self,
        hv: UnsafePointer[Self.HitsView, MutAnyOrigin],
        nhits: UInt32,
        maxNumberOfTuples: UInt32,
        cudaStream: cudaStream_t,
    ):
        pass

    # C++: HelixFitOnGPU::launchRiemannKernelsOnCPU (RiemannFitOnGPU.cc)
    fn launchRiemannKernelsOnCPU(
        self,
        hv: UnsafePointer[Self.HitsView, MutAnyOrigin],
        nhits: UInt32,
        maxNumberOfTuples: UInt32,
    ) raises:
        debug_assert(Bool(self.tuples_))

        var nConcurrent = Int(Self.maxNumberOfConcurrentFits_)
        # C++ sizes these in bytes
        # (maxNumberOfConcurrentFits_ * sizeof(riemannFit::Matrix3xNd<4>) /
        # sizeof(double), etc.); ported as element counts.
        var hitsGPU_ = List[Float64](length=nConcurrent * 3 * 4, fill=0.0)
        var hits_geGPU_ = List[Float32](length=nConcurrent * 6 * 4, fill=0.0)
        var fast_fit_resultsGPU_ = List[Float64](
            length=nConcurrent * 4, fill=0.0
        )
        var circle_fit_resultsGPU_ = List[riemannFitUtils.CircleFit](
            length=nConcurrent, fill=riemannFitUtils.CircleFit()
        )

        var offset: UInt32 = 0
        while offset < maxNumberOfTuples:
            # triplets
            kernelFastFit[3](
                self.tuples_, self.tupleMultiplicity_, 3, hv,
                hitsGPU_.unsafe_ptr(), hits_geGPU_.unsafe_ptr(),
                fast_fit_resultsGPU_.unsafe_ptr(), offset,
            )
            kernelCircleFit[3](
                self.tupleMultiplicity_, 3, Float64(self.bField_),
                hitsGPU_.unsafe_ptr(), hits_geGPU_.unsafe_ptr(),
                fast_fit_resultsGPU_.unsafe_ptr(),
                circle_fit_resultsGPU_.unsafe_ptr(), offset,
            )
            kernelLineFit[3](
                self.tupleMultiplicity_, 3, Float64(self.bField_),
                self.outputSoa_, hitsGPU_.unsafe_ptr(),
                hits_geGPU_.unsafe_ptr(), fast_fit_resultsGPU_.unsafe_ptr(),
                circle_fit_resultsGPU_.unsafe_ptr(), offset,
            )

            # quads
            kernelFastFit[4](
                self.tuples_, self.tupleMultiplicity_, 4, hv,
                hitsGPU_.unsafe_ptr(), hits_geGPU_.unsafe_ptr(),
                fast_fit_resultsGPU_.unsafe_ptr(), offset,
            )
            kernelCircleFit[4](
                self.tupleMultiplicity_, 4, Float64(self.bField_),
                hitsGPU_.unsafe_ptr(), hits_geGPU_.unsafe_ptr(),
                fast_fit_resultsGPU_.unsafe_ptr(),
                circle_fit_resultsGPU_.unsafe_ptr(), offset,
            )
            kernelLineFit[4](
                self.tupleMultiplicity_, 4, Float64(self.bField_),
                self.outputSoa_, hitsGPU_.unsafe_ptr(),
                hits_geGPU_.unsafe_ptr(), fast_fit_resultsGPU_.unsafe_ptr(),
                circle_fit_resultsGPU_.unsafe_ptr(), offset,
            )

            if self.fit5as4_:
                # penta (only first 4)
                kernelFastFit[4](
                    self.tuples_, self.tupleMultiplicity_, 5, hv,
                    hitsGPU_.unsafe_ptr(), hits_geGPU_.unsafe_ptr(),
                    fast_fit_resultsGPU_.unsafe_ptr(), offset,
                )
                kernelCircleFit[4](
                    self.tupleMultiplicity_, 5, Float64(self.bField_),
                    hitsGPU_.unsafe_ptr(), hits_geGPU_.unsafe_ptr(),
                    fast_fit_resultsGPU_.unsafe_ptr(),
                    circle_fit_resultsGPU_.unsafe_ptr(), offset,
                )
                kernelLineFit[4](
                    self.tupleMultiplicity_, 5, Float64(self.bField_),
                    self.outputSoa_, hitsGPU_.unsafe_ptr(),
                    hits_geGPU_.unsafe_ptr(),
                    fast_fit_resultsGPU_.unsafe_ptr(),
                    circle_fit_resultsGPU_.unsafe_ptr(), offset,
                )
            else:
                # penta (all 5)
                kernelFastFit[5](
                    self.tuples_, self.tupleMultiplicity_, 5, hv,
                    hitsGPU_.unsafe_ptr(), hits_geGPU_.unsafe_ptr(),
                    fast_fit_resultsGPU_.unsafe_ptr(), offset,
                )
                kernelCircleFit[5](
                    self.tupleMultiplicity_, 5, Float64(self.bField_),
                    hitsGPU_.unsafe_ptr(), hits_geGPU_.unsafe_ptr(),
                    fast_fit_resultsGPU_.unsafe_ptr(),
                    circle_fit_resultsGPU_.unsafe_ptr(), offset,
                )
                kernelLineFit[5](
                    self.tupleMultiplicity_, 5, Float64(self.bField_),
                    self.outputSoa_, hitsGPU_.unsafe_ptr(),
                    hits_geGPU_.unsafe_ptr(),
                    fast_fit_resultsGPU_.unsafe_ptr(),
                    circle_fit_resultsGPU_.unsafe_ptr(), offset,
                )

            offset += Self.maxNumberOfConcurrentFits_

    # C++: HelixFitOnGPU::launchBrokenLineKernelsOnCPU (BrokenLineFitOnGPU.cc)
    fn launchBrokenLineKernelsOnCPU(
        self,
        hv: UnsafePointer[Self.HitsView, MutAnyOrigin],
        nhits: UInt32,
        maxNumberOfTuples: UInt32,
    ) raises:
        debug_assert(Bool(self.tuples_))

        var nConcurrent = Int(Self.maxNumberOfConcurrentFits_)
        var hitsGPU_ = List[Float64](length=nConcurrent * 3 * 4, fill=0.0)
        var hits_geGPU_ = List[Float32](length=nConcurrent * 6 * 4, fill=0.0)
        var fast_fit_resultsGPU_ = List[Float64](
            length=nConcurrent * 4, fill=0.0
        )

        var offset: UInt32 = 0
        while offset < maxNumberOfTuples:
            # fit triplets
            kernelBLFastFit[3](
                self.tuples_, self.tupleMultiplicity_, hv,
                hitsGPU_.unsafe_ptr(), hits_geGPU_.unsafe_ptr(),
                fast_fit_resultsGPU_.unsafe_ptr(), 3, offset,
            )
            kernelBLFit[3](
                self.tupleMultiplicity_, Float64(self.bField_),
                self.outputSoa_, hitsGPU_.unsafe_ptr(),
                hits_geGPU_.unsafe_ptr(), fast_fit_resultsGPU_.unsafe_ptr(),
                3, offset,
            )

            # fit quads
            kernelBLFastFit[4](
                self.tuples_, self.tupleMultiplicity_, hv,
                hitsGPU_.unsafe_ptr(), hits_geGPU_.unsafe_ptr(),
                fast_fit_resultsGPU_.unsafe_ptr(), 4, offset,
            )
            kernelBLFit[4](
                self.tupleMultiplicity_, Float64(self.bField_),
                self.outputSoa_, hitsGPU_.unsafe_ptr(),
                hits_geGPU_.unsafe_ptr(), fast_fit_resultsGPU_.unsafe_ptr(),
                4, offset,
            )

            if self.fit5as4_:
                # fit penta (only first 4)
                kernelBLFastFit[4](
                    self.tuples_, self.tupleMultiplicity_, hv,
                    hitsGPU_.unsafe_ptr(), hits_geGPU_.unsafe_ptr(),
                    fast_fit_resultsGPU_.unsafe_ptr(), 5, offset,
                )
                kernelBLFit[4](
                    self.tupleMultiplicity_, Float64(self.bField_),
                    self.outputSoa_, hitsGPU_.unsafe_ptr(),
                    hits_geGPU_.unsafe_ptr(),
                    fast_fit_resultsGPU_.unsafe_ptr(), 5, offset,
                )
            else:
                # fit penta (all 5)
                kernelBLFastFit[5](
                    self.tuples_, self.tupleMultiplicity_, hv,
                    hitsGPU_.unsafe_ptr(), hits_geGPU_.unsafe_ptr(),
                    fast_fit_resultsGPU_.unsafe_ptr(), 5, offset,
                )
                kernelBLFit[5](
                    self.tupleMultiplicity_, Float64(self.bField_),
                    self.outputSoa_, hitsGPU_.unsafe_ptr(),
                    hits_geGPU_.unsafe_ptr(),
                    fast_fit_resultsGPU_.unsafe_ptr(), 5, offset,
                )

            offset += Self.maxNumberOfConcurrentFits_

    fn allocateOnGPU(
        mut self,
        tuples: UnsafePointer[Self.Tuples, MutAnyOrigin],
        tupleMultiplicity: UnsafePointer[Self.TupleMultiplicity, MutAnyOrigin],
        outputSoA: UnsafePointer[Self.OutputSoA, MutAnyOrigin],
    ):
        self.tuples_ = tuples
        self.tupleMultiplicity_ = tupleMultiplicity
        self.outputSoa_ = outputSoA
        debug_assert(Bool(self.tuples_))
        debug_assert(Bool(self.tupleMultiplicity_))
        debug_assert(Bool(self.outputSoa_))

    fn deallocateOnGPU(mut self):
        pass
