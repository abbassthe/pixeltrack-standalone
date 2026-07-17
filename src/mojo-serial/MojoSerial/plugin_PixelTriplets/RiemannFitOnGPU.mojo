import math

import CAConstants
import RiemannFit
from FitUtils import Rfit as FitRfit
from HelixFitOnGPU import Rfit
from MojoSerial.CUDADataFormats.PixelTrackHeterogeneous import (
    PixelTrack as pixelTrack,
)
from MojoSerial.CUDADataFormats.TrackingRecHit2DSOAView import (
    TrackingRecHit2DSOAView,
)


alias RIEMANN_DEBUG = False


alias HitsOnGPU = TrackingRecHit2DSOAView
alias Tuples = pixelTrack.HitContainer
alias OutputSoA = pixelTrack.TrackSoA
alias CircleFit = FitRfit.circle_fit
alias LineFit = FitRfit.line_fit

fn kernelFastFit[N: Int](
    foundNtuplets: UnsafePointer[Tuples],
    tupleMultiplicity: UnsafePointer[CAConstants.TupleMultiplicity],
    nHits: UInt32,
    hhp: UnsafePointer[HitsOnGPU],
    phits: UnsafePointer[Float64],
    phits_ge: UnsafePointer[Float32],
    pfast_fit: UnsafePointer[Float64],
    offset: UInt32,
):
    comptime hitsInFit: UInt32 = N

    assert(hitsInFit <= nHits)

    assert(pfast_fit)
    assert(foundNtuplets)
    assert(tupleMultiplicity)

    var local_start = 0

    @parameter
    if RIEMANN_DEBUG:
        if local_start == 0:
            print(
                f"{tupleMultiplicity[].size(nHits)} Ntuple of size {nHits} for {hitsInFit} hits to fit",
            )

    var local_idx: Int = local_start
    var nt = Rfit.maxNumberOfConcurrentFits().cast[Int]()
    while local_idx < nt:
        var tuple_idx = local_idx + offset.cast[Int]()
        if tuple_idx >= tupleMultiplicity[].size(nHits).cast[Int]():
            break

        var tkid = (tupleMultiplicity[].begin(nHits) + tuple_idx)[]
        assert(tkid < foundNtuplets[].nbins())
        assert(foundNtuplets[].size(tkid) == nHits)

        var hits = Rfit.Map3xNd[N](phits + local_idx)
        var fast_fit = Rfit.Map4d(pfast_fit + local_idx)
        var hits_ge = Rfit.Map6xNf[N](phits_ge + local_idx)

        var hitId = foundNtuplets[].begin(tkid)
        var i: UInt32 = 0
        while i < hitsInFit:
            var idx = i.cast[Int]()
            var hit = hitId[idx]
            var ge = InlineArray[Float32, 6](fill=0)
            hhp[].cpeParams().detParams(hhp[].detectorIndex(hit)).frame.toGlobal(
                hhp[].xerrLocal(hit),
                0,
                hhp[].yerrLocal(hit),
                ge.unsafe_ptr(),
            )

            hits[0, idx] = hhp[].xGlobal(hit)
            hits[1, idx] = hhp[].yGlobal(hit)
            hits[2, idx] = hhp[].zGlobal(hit)
            hits_ge[0, idx] = ge[0]
            hits_ge[1, idx] = ge[1]
            hits_ge[2, idx] = ge[2]
            hits_ge[3, idx] = ge[3]
            hits_ge[4, idx] = ge[4]
            hits_ge[5, idx] = ge[5]
            i += 1

        RiemannFit.Fast_fit(hits, fast_fit)

        assert(fast_fit[0] == fast_fit[0])
        assert(fast_fit[1] == fast_fit[1])
        assert(fast_fit[2] == fast_fit[2])
        assert(fast_fit[3] == fast_fit[3])

        local_idx += 1

fn kernelCircleFit[N: Int](
    tupleMultiplicity: UnsafePointer[CAConstants.TupleMultiplicity],
    nHits: UInt32,
    B: Float64,
    phits: UnsafePointer[Float64],
    phits_ge: UnsafePointer[Float32],
    pfast_fit_input: UnsafePointer[Float64],
    circle_fit: UnsafePointer[CircleFit],
    offset: UInt32,
):
    assert(circle_fit)
    assert(N <= nHits)

    var local_start = 0
    var local_idx: Int = local_start
    var nt = Rfit.maxNumberOfConcurrentFits().cast[Int]()
    while local_idx < nt:
        var tuple_idx = local_idx + offset.cast[Int]()
        if tuple_idx >= tupleMultiplicity[].size(nHits).cast[Int]():
            break

        var hits = Rfit.Map3xNd[N](phits + local_idx)
        var fast_fit = Rfit.Map4d(pfast_fit_input + local_idx)
        var hits_ge = Rfit.Map6xNf[N](phits_ge + local_idx)

        var rad = FitRfit.VectorNd[N]()
        var i: Int = 0
        while i < N:
            var x = hits[0, i]
            var y = hits[1, i]
            rad[i] = sqrt(x * x + y * y)
            i += 1

        var hits_cov = FitRfit.Matrix2Nd[N].Zero()
        FitRfit.loadCovariance2D(hits_ge, hits_cov)

        circle_fit[local_idx] = RiemannFit.Circle_fit(
            hits.block(0, 0, 2, N),
            hits_cov,
            fast_fit,
            rad,
            B,
            True,
        )

        @parameter
        if RIEMANN_DEBUG:
            # let tkid = (tupleMultiplicity[].begin(nHits) + tuple_idx)[]
            # print(
            #     f"kernelCircleFit circle.par(0,1,2): {tkid} {circle_fit[local_idx].par[0]},{circle_fit[local_idx].par[1]},{circle_fit[local_idx].par[2]}",
            # )

        local_idx += 1


fn kernelLineFit[N: Int](
    tupleMultiplicity: UnsafePointer[CAConstants.TupleMultiplicity],
    nHits: UInt32,
    B: Float64,
    results: UnsafePointer[OutputSoA],
    phits: UnsafePointer[Float64],
    phits_ge: UnsafePointer[Float32],
    pfast_fit_input: UnsafePointer[Float64],
    circle_fit: UnsafePointer[CircleFit],
    offset: UInt32,
):
    assert(results)
    assert(circle_fit)
    assert(N <= nHits)

    var local_start = 0
    var local_idx: Int = local_start
    var nt = Rfit.maxNumberOfConcurrentFits().cast[Int]()
    var tuples_for_size = tupleMultiplicity[].size(nHits).cast[Int]()
    while local_idx < nt:
        var tuple_idx = local_idx + offset.cast[Int]()
        if tuple_idx >= tuples_for_size:
            break

        var tkid = (tupleMultiplicity[].begin(nHits) + tuple_idx)[]

        var hits = Rfit.Map3xNd[N](phits + local_idx)
        var fast_fit = Rfit.Map4d(pfast_fit_input + local_idx)
        var hits_ge = Rfit.Map6xNf[N](phits_ge + local_idx)

        ref line_fit = RiemannFit.Line_fit(
            hits,
            hits_ge,
            circle_fit[local_idx],
            fast_fit,
            B,
            True,
        )

        FitRfit.fromCircleToPerigee(circle_fit[local_idx])

        var track_idx = tkid.cast[Int]()
        results[].stateAtBS.copyFromCircle(
            circle_fit[local_idx].par,
            circle_fit[local_idx].cov,
            line_fit.par,
            line_fit.cov,
            Float32(1.0 / B),
            track_idx.cast[Int32](),
        )
        results[].pt[track_idx] = Float32(B) / Float32(
            abs(circle_fit[local_idx].par[2])
        )
        results[].eta[track_idx] = Float32(math.asinh(line_fit.par[0]))
        var chi2 = Float64(circle_fit[local_idx].chi2) + line_fit.chi2
        results[].chi2[track_idx] = Float32(
            chi2 / Float64(2 * N - 5)
        )

        @parameter
        if RIEMANN_DEBUG:
            print(
                f"kernelLineFit size {N} for {nHits} hits circle.par(0,1,2): {tkid} {circle_fit[local_idx].par[0]},{circle_fit[local_idx].par[1]},{circle_fit[local_idx].par[2]}",
            )
            print(
                f"kernelLineFit line.par(0,1): {tkid} {line_fit.par[0]},{line_fit.par[1]}",
            )
            print(
                f"kernelLineFit chi2 cov {circle_fit[local_idx].chi2}/{line_fit.chi2} {circle_fit[local_idx].cov[0, 0]},{circle_fit[local_idx].cov[1, 1]},{circle_fit[local_idx].cov[2, 2]},{line_fit.cov[0, 0]},{line_fit.cov[1, 1]}",
            )

        local_idx += 1
