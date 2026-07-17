import math
import CAConstants
import BrokenLine
from HelixFitOnGPU import Rfit
from MojoSerial.CUDADataFormats.TrackingRecHit2DSOAView import TrackingRecHit2DSOAView
from MojoSerial.CUDADataFormats.PixelTrackHeterogeneous import PixelTrack as pixelTrack
from sync import AtomicInt


alias HitsOnGPU = TrackingRecHit2DSOAView
alias Tuples = pixelTrack.HitContainer
alias OutputSoA = pixelTrack.TrackSoA

alias BROKENLINE_DEBUG = False

alias BL_DUMP_HITS = False


fn kernelBLFastFit[N: Int](
    foundNtuplets: UnsafePointer[Tuples],
    tupleMultiplicity: UnsafePointer[CAConstants.TupleMultiplicity],
    hhp: UnsafePointer[HitsOnGPU],
    phits: UnsafePointer[Float64],
    phits_ge: UnsafePointer[Float32],
    pfast_fit: UnsafePointer[Float64],
    nHits: UInt32,
    offset: UInt32,
):
    var hitsInFit: UInt32 = N
    assert(hitsInFit <= nHits)

    assert(hhp)
    assert(pfast_fit)
    assert(foundNtuplets)
    assert(tupleMultiplicity)

    var local_start = 0

    @parameter
    if BROKENLINE_DEBUG:
        if local_start == 0:
            print(
                f"{foundNtuplets[].nbins()} total Ntuple",
            )
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

        @parameter
        if BL_DUMP_HITS:
            var done = AtomicInt(0)
            var dump = (
                foundNtuplets[].size(tkid) == 5 and done.fetch_add(1) == 0
            )

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

            @parameter
            if BL_DUMP_HITS:
                if dump:
                    print(
                        f"Hit global: {tkid}: {hhp[].detectorIndex(hit)} hits.col({i}) << {hhp[].xGlobal(hit)},{hhp[].yGlobal(hit)},{hhp[].zGlobal(hit)}",
                    )
                    print(
                        f"Error: {tkid}: {hhp[].detectorIndex(hit)}  hits_ge.col({i}) << {ge[0]},{ge[1]},{ge[2]},{ge[3]},{ge[4]},{ge[5]}",
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

        BrokenLine.BL_Fast_fit(hits, fast_fit)


        assert(fast_fit[0] == fast_fit[0])
        assert(fast_fit[1] == fast_fit[1])
        assert(fast_fit[2] == fast_fit[2])
        assert(fast_fit[3] == fast_fit[3])

        local_idx += 1


fn kernelBLFit[N: Int](
    tupleMultiplicity: UnsafePointer[CAConstants.TupleMultiplicity],
    B: Float64,
    results: UnsafePointer[OutputSoA],
    phits: UnsafePointer[Float64],
    phits_ge: UnsafePointer[Float32],
    pfast_fit: UnsafePointer[Float64],
    nHits: UInt32,
    offset: UInt32,
):
    assert(N <= nHits)

    assert(results)
    assert(pfast_fit)

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
        var fast_fit = Rfit.Map4d(pfast_fit + local_idx)
        var hits_ge = Rfit.Map6xNf[N](phits_ge + local_idx)

        var data = BrokenLine.PreparedBrokenLineData[N]()
        var Jacob = Rfit.Matrix3d()
        var circle = BrokenLine.karimaki_circle_fit()
        var line = Rfit.line_fit()

        BrokenLine.prepareBrokenLineData(hits, fast_fit, B, data)
        BrokenLine.BL_Line_fit(hits_ge, fast_fit, B, data, line)
        BrokenLine.BL_Circle_fit(hits, hits_ge, fast_fit, B, data, circle)

        var track_idx = tkid.cast[Int]()
        results[].stateAtBS.copyFromCircle(
            circle.par,
            circle.cov,
            line.par,
            line.cov,
            Float32(1.0 / B),
            track_idx.cast[Int32](),
        )
        results[].pt[track_idx] = Float32(B) / Float32(abs(circle.par[2]))
        results[].eta[track_idx] = Float32(math.asinh(line.par[0]))
        var chi2 = Float64(circle.chi2) + line.chi2
        results[].chi2[track_idx] = Float32(
            chi2 / Float64(2 * N - 5)
        )

        @parameter
        if BROKENLINE_DEBUG:
            if not (circle.chi2 >= 0) or not (line.chi2 >= 0):
                print(
                    f"kernelBLFit failed! {circle.chi2}/{line.chi2}",
                )
            print(
                f"kernelBLFit size {N} for {nHits} hits circle.par(0,1,2): {tkid} {circle.par[0]},{circle.par[1]},{circle.par[2]}",
            )
            print(
                f"kernelBLHits line.par(0,1): {tkid} {line.par[0]},{line.par[1]}",
            )
            print(
                f"kernelBLHits chi2 cov {circle.chi2}/{line.chi2}  {circle.cov[0, 0]},{circle.cov[1, 1]},{circle.cov[2, 2]},{line.cov[0, 0]},{line.cov[1, 1]}",
            )

        local_idx += 1
