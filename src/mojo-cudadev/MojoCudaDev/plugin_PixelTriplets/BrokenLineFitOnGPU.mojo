# Mojo port of plugin-PixelTriplets/BrokenLineFitOnGPU.h.
#
# BROKENLINE_DEBUG / BL_DUMP_HITS are the C++ #ifdefs: pass -D BROKENLINE_DEBUG
# (or -D BL_DUMP_HITS) to mojo build/run to enable.
#
# On host, block_dim/grid_dim collapse to 1 and block_idx/thread_idx to 0,
# reproducing cudaCompat.h so launchBrokenLineKernelsOnCPU runs the same body.
import math
from memory import stack_allocation
from std.memory import AddressSpace
from sys import is_gpu
from std.sys.param_env import is_defined

from MojoCudaDev.CUDACore.CUDAAtomics import atomic_fetch_add
from MojoCudaDev.CUDADataFormats.TrackingRecHit2DSOAView import (
    TrackingRecHit2DSOAView,
)
from MojoCudaDev.CUDADataFormats.TrackSoAHeterogeneousT import pixelTrack
from MojoCudaDev.MojoBridge.Matrix import to_layout_tensor
from MojoCudaDev.plugin_PixelTriplets.CAConstants import caConstants
from MojoCudaDev.plugin_PixelTriplets.FitUtils import (
    riemannFit as riemannFitUtils,
)
from MojoCudaDev.plugin_PixelTriplets.HelixFitOnGpu import riemannFit
import MojoCudaDev.plugin_PixelTriplets.BrokenLine as brokenline

comptime HitsOnGPU = TrackingRecHit2DSOAView
comptime Tuples = pixelTrack.HitContainer
comptime OutputSoA = pixelTrack.TrackSoA

comptime BROKENLINE_DEBUG = is_defined["BROKENLINE_DEBUG"]()
comptime BL_DUMP_HITS = is_defined["BL_DUMP_HITS"]()

comptime FIRST = 0
comptime STRIDE = 1


@always_inline
fn _grid_stride() -> Tuple[Int, Int]:
    @parameter
    if is_gpu():
        from std.gpu import thread_idx, block_idx, block_dim, grid_dim

        return (
            Int(block_dim.x) * Int(block_idx.x) + Int(thread_idx.x),
            Int(grid_dim.x) * Int(block_dim.x),
        )
    else:
        return (Int(0), Int(1))


# C++: `__shared__ int done; done = 0; __syncthreads();
#       bool dump = (size == 5 && 0 == atomicAdd(&done, 1));`
# Picks exactly one thread per block, re-armed on every loop iteration.
@always_inline
fn _dump_this_iteration(is_five_hit: Bool) -> Bool:
    @parameter
    if is_gpu():
        from std.gpu.sync import barrier

        var done = stack_allocation[
            1, Int32, address_space = AddressSpace.SHARED
        ]()
        done[0] = 0
        barrier()
        return is_five_hit and atomic_fetch_add(done, Int32(1)) == 0
    else:
        return is_five_hit


# C++: kernel_BLFastFit<N> -- BrokenLineFitOnGPU.h:26-116
fn kernelBLFastFit[
    N: Int
](
    foundNtuplets: UnsafePointer[Tuples, MutAnyOrigin],
    tupleMultiplicity: UnsafePointer[
        caConstants.TupleMultiplicity, MutAnyOrigin
    ],
    hhp: UnsafePointer[HitsOnGPU, MutAnyOrigin],
    phits: UnsafePointer[Float64, MutAnyOrigin],
    phits_ge: UnsafePointer[Float32, MutAnyOrigin],
    pfast_fit: UnsafePointer[Float64, MutAnyOrigin],
    nHits: UInt32,
    offset: UInt32,
):
    comptime hitsInFit = N

    debug_assert(UInt32(hitsInFit) <= nHits)
    debug_assert(Bool(hhp))
    debug_assert(Bool(pfast_fit))
    debug_assert(Bool(foundNtuplets))
    debug_assert(Bool(tupleMultiplicity))

    var grid = _grid_stride()
    var local_start = grid[FIRST]
    var stride = grid[STRIDE]

    @parameter
    if BROKENLINE_DEBUG:
        if local_start == 0:
            print(foundNtuplets[].nOnes(), "total Ntuple")
            print(
                tupleMultiplicity[].size(Int(nHits)),
                "Ntuple of size",
                nHits,
                "for",
                hitsInFit,
                "hits to fit",
            )

    var nt = Int(riemannFit.maxNumberOfConcurrentFits)
    var local_idx = local_start
    while local_idx < nt:
        var tuple_idx = local_idx + Int(offset)
        if tuple_idx >= Int(tupleMultiplicity[].size(Int(nHits))):
            break

        # get it from the ntuple container (one to one to helix)
        var tkid = UInt32((tupleMultiplicity[].begin(Int(nHits)) + tuple_idx)[])
        debug_assert(Int(tkid) < foundNtuplets[].nOnes())
        debug_assert(foundNtuplets[].size(Int(tkid)) == nHits)

        var hits = riemannFit.Map3xNd[N](phits + local_idx)
        var fast_fit = riemannFit.Map4d(pfast_fit + local_idx)
        var hits_ge = riemannFit.Map6xNf[N](phits_ge + local_idx)

        var dump = False

        @parameter
        if BL_DUMP_HITS:
            dump = _dump_this_iteration(foundNtuplets[].size(Int(tkid)) == 5)

        # Prepare data structure
        var hitId = foundNtuplets[].begin(Int(tkid))
        for i in range(hitsInFit):
            var hit = Int(hitId[i])
            var ge = InlineArray[Float32, 6](fill=0)
            hhp[].cpeParams().detParams(
                Int32(hhp[].detectorIndex(hit))
            ).frame.toGlobal(
                hhp[].xerrLocal(hit),
                0,
                hhp[].yerrLocal(hit),
                ge.unsafe_ptr(),
            )

            @parameter
            if BL_DUMP_HITS:
                if dump:
                    print(
                        "Hit global:",
                        tkid,
                        ":",
                        hhp[].detectorIndex(hit),
                        "hits.col(",
                        i,
                        ") <<",
                        hhp[].xGlobal(hit),
                        ",",
                        hhp[].yGlobal(hit),
                        ",",
                        hhp[].zGlobal(hit),
                    )
                    print(
                        "Error:",
                        tkid,
                        ":",
                        hhp[].detectorIndex(hit),
                        " hits_ge.col(",
                        i,
                        ") <<",
                        ge[0],
                        ",",
                        ge[1],
                        ",",
                        ge[2],
                        ",",
                        ge[3],
                        ",",
                        ge[4],
                        ",",
                        ge[5],
                    )

            hits[0, i] = hhp[].xGlobal(hit).cast[DType.float64]()
            hits[1, i] = hhp[].yGlobal(hit).cast[DType.float64]()
            hits[2, i] = hhp[].zGlobal(hit).cast[DType.float64]()
            hits_ge[0, i] = ge[0]
            hits_ge[1, i] = ge[1]
            hits_ge[2, i] = ge[2]
            hits_ge[3, i] = ge[3]
            hits_ge[4, i] = ge[4]
            hits_ge[5, i] = ge[5]

        brokenline.fastFit(hits, fast_fit)

        # no NaN here....
        debug_assert(fast_fit[0, 0] == fast_fit[0, 0])
        debug_assert(fast_fit[1, 0] == fast_fit[1, 0])
        debug_assert(fast_fit[2, 0] == fast_fit[2, 0])
        debug_assert(fast_fit[3, 0] == fast_fit[3, 0])

        local_idx += stride


# C++: kernel_BLFit<N> -- BrokenLineFitOnGPU.h:118-184
fn kernelBLFit[
    N: Int
](
    tupleMultiplicity: UnsafePointer[
        caConstants.TupleMultiplicity, MutAnyOrigin
    ],
    bField: Float64,
    results: UnsafePointer[OutputSoA, MutAnyOrigin],
    phits: UnsafePointer[Float64, MutAnyOrigin],
    phits_ge: UnsafePointer[Float32, MutAnyOrigin],
    pfast_fit: UnsafePointer[Float64, MutAnyOrigin],
    nHits: UInt32,
    offset: UInt32,
):
    debug_assert(UInt32(N) <= nHits)
    debug_assert(Bool(results))
    debug_assert(Bool(pfast_fit))

    var grid = _grid_stride()
    var local_start = grid[FIRST]
    var stride = grid[STRIDE]

    var nt = Int(riemannFit.maxNumberOfConcurrentFits)
    var local_idx = local_start
    while local_idx < nt:
        var tuple_idx = local_idx + Int(offset)
        if tuple_idx >= Int(tupleMultiplicity[].size(Int(nHits))):
            break

        # get it for the ntuple container (one to one to helix)
        var tkid = UInt32((tupleMultiplicity[].begin(Int(nHits)) + tuple_idx)[])

        var hits = riemannFit.Map3xNd[N](phits + local_idx)
        var fast_fit = riemannFit.Map4d(pfast_fit + local_idx)
        var hits_ge = riemannFit.Map6xNf[N](phits_ge + local_idx)

        var data = brokenline.PreparedBrokenLineData[N]()
        var circle = brokenline.karimaki_circle_fit()
        var line = riemannFitUtils.LineFit()

        brokenline.prepareBrokenLineData(hits, fast_fit, bField, data)
        brokenline.lineFit(hits_ge, fast_fit, bField, data, line)
        brokenline.circleFit(hits, hits_ge, fast_fit, bField, data, circle)

        var track_idx = Int(tkid)
        var cp_buf = InlineArray[Scalar[DType.float64], 3](uninitialized=True)
        var ccov_buf = InlineArray[Scalar[DType.float64], 9](uninitialized=True)
        var lp_buf = InlineArray[Scalar[DType.float64], 2](uninitialized=True)
        var lcov_buf = InlineArray[Scalar[DType.float64], 4](uninitialized=True)
        results[].stateAtBS.copyFromCircle(
            to_layout_tensor(circle.par, cp_buf),
            to_layout_tensor(circle.cov, ccov_buf),
            to_layout_tensor(line.par, lp_buf),
            to_layout_tensor(line.cov, lcov_buf),
            Float32(1.0 / bField),
            Int32(track_idx),
        )
        results[].pt[track_idx] = Float32(bField) / Float32(
            abs(circle.par[2, 0])
        )
        results[].eta[track_idx] = Float32(math.asinh(line.par[0, 0]))
        results[].chi2[track_idx] = Float32(
            (Float64(circle.chi2) + line.chi2) / Float64(2 * N - 5)
        )

        @parameter
        if BROKENLINE_DEBUG:
            if not (circle.chi2 >= 0) or not (line.chi2 >= 0):
                print("kernelBLFit failed!", circle.chi2, "/", line.chi2)
            print(
                "kernelBLFit size",
                N,
                "for",
                nHits,
                "hits circle.par(0,1,2):",
                tkid,
                circle.par[0, 0],
                ",",
                circle.par[1, 0],
                ",",
                circle.par[2, 0],
            )
            print(
                "kernelBLHits line.par(0,1):",
                tkid,
                line.par[0, 0],
                ",",
                line.par[1, 0],
            )
            print(
                "kernelBLHits chi2 cov",
                circle.chi2,
                "/",
                line.chi2,
                " ",
                circle.cov[0, 0],
                ",",
                circle.cov[1, 1],
                ",",
                circle.cov[2, 2],
                ",",
                line.cov[0, 0],
                ",",
                line.cov[1, 1],
            )

        local_idx += stride
