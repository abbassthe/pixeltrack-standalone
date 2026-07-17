import CAConstants
import GPUCACell
from sys import is_defined
from MojoSerial.CUDACore.CUDACompat import CUDACompat
from MojoSerial.CUDADataFormats.TrackingRecHit2DSOAView import (
    TrackingRecHit2DSOAView,
    Hist,
)
from MojoSerial.DataFormats.ApproxAtan2 import ApproxAtan2

alias CellNeighbors = CAConstants.CellNeighbors
alias CellTracks = CAConstants.CellTracks
alias CellNeighborsVector = CAConstants.CellNeighborsVector
alias CellTracksVector = CAConstants.CellTracksVector


fn doubletsFromHisto(
    layerPairs: UnsafePointer[UInt8],
    nPairs: UInt32,
    cells: UnsafePointer[GPUCACell],
    nCells: UnsafePointer[UInt32],
    cellNeighbors: UnsafePointer[CellNeighborsVector],
    cellTracks: UnsafePointer[CellTracksVector],
    hh: TrackingRecHit2DSOAView,
    isOuterHitOfCell: UnsafePointer[GPUCACell.OuterHitOfCell],
    phicuts: UnsafePointer[Int16],
    minz: UnsafePointer[Float32],
    maxz: UnsafePointer[Float32],
    maxr: UnsafePointer[Float32],
    ideal_cond: Bool,
    doClusterCut: Bool,
    doZ0Cut: Bool,
    doPtCut: Bool,
    maxNumOfDoublets: UInt32,
):
    # ysize cuts (z in the barrel) times 8
    # these are used if doClusterCut is true
    comptime minYsizeB1: Int = 36
    comptime minYsizeB2: Int = 28
    comptime maxDYsize12: Int = 28
    comptime maxDYsize: Int = 20
    comptime maxDYPred: Int = 20
    comptime dzdrFact: Float32 = 8.0 * 0.0285 / 0.015  # from dz/dr to "DY"

    var isOuterLadder: Bool = ideal_cond

    ref hist = hh.phiBinner()
    var offsets : UnsafePointer[UInt32] = hh.hitsLayerStart()
    assert(offsets)

    fn layerSize(li: UInt8) -> UInt32:
        var idx = li.cast[Int]()
        return offsets[idx + 1] - offsets[idx]

    # nPairsMax to be optimized later (originally was 64).
    # If it should be much bigger, consider using a block-wide parallel prefix scan,
    # e.g. see https://nvlabs.github.io/cub/classcub_1_1_warp_scan.html
    comptime nPairsMax: Int = CAConstants.maxNumberOfLayerPairs().cast[Int]()
    assert(nPairs <= nPairsMax.cast[UInt32]())
    var innerLayerCumulativeSize = InlineArray[UInt32, nPairsMax]()
    var ntot: UInt32 = 0

    innerLayerCumulativeSize[0] = layerSize(layerPairs[0])
    for i in range(1, nPairs.cast[Int]()):
        innerLayerCumulativeSize[i] = (
            innerLayerCumulativeSize[i - 1]
            + layerSize(layerPairs[2 * i])
        )
    ntot = innerLayerCumulativeSize[nPairs.cast[Int]() - 1]

    # x runs faster
    var idy = 0
    var first: UInt32 = 0
    var stride = 1

    var pairLayerId: UInt32 = 0  # cannot go backward
    var j: UInt32 = idy
    while j < ntot:
        while j >= innerLayerCumulativeSize[pairLayerId.cast[Int]()]:
            pairLayerId += 1

        assert(pairLayerId < nPairs)
        assert(j < innerLayerCumulativeSize[pairLayerId.cast[Int]()])
        if pairLayerId > 0:
            assert(j >= innerLayerCumulativeSize[(pairLayerId - 1).cast[Int]()])
        else:
            assert(0 == pairLayerId)

        var inner :UInt8 = layerPairs[(2 * pairLayerId).cast[Int]()]
        var outer :UInt8= layerPairs[(2 * pairLayerId + 1).cast[Int]()]
        assert(outer > inner)
        var inner_idx = inner.cast[Int]()
        var outer_idx = outer.cast[Int]()

        var hoff = Hist.histOff(outer.cast[UInt32]())

        var i = j
        if pairLayerId != 0:
            i -= innerLayerCumulativeSize[(pairLayerId - 1).cast[Int]()]
        i += offsets[inner_idx]

        # printf("Hit in Layer %d %d %d %d\n", i, inner, pairLayerId, j);

        assert(i >= offsets[inner_idx])
        assert(i < offsets[inner_idx + 1])

        # found hit corresponding to our cuda thread, now do the job
        var mi = hh.detectorIndex(i.cast[Int]())
        if mi > 2000:
            j += 1
            continue  # invalid

        # maybe clever, not effective when zoCut is on
        # auto bpos = (mi%8)/4  # if barrel is 1 for z>0
        # auto fpos = (outer>3) & (outer<7)
        # if ((inner<3) & (outer>3)) and bpos!=fpos: continue

        var mez = hh.zGlobal(i.cast[Int]())

        if mez < minz[pairLayerId.cast[Int]()] or mez > maxz[pairLayerId.cast[Int]()]:
            j += 1
            continue

        var mes: Int16 = -1
        if doClusterCut:
            # if ideal treat inner ladder as outer
            if inner == 0:
                assert(mi < 96)
            if ideal_cond:
                isOuterLadder = True
            else:
                isOuterLadder = ((mi / 8).cast[UInt32]() % 2) == 0

            # in any case we always test mes>0 ...
            if inner > 0 or isOuterLadder:
                mes = hh.clusterSizeY(i.cast[Int]())
            else:
                mes = -1

            if inner == 0 and outer > 3:  # B1 and F1
                var mes_i = mes.cast[Int]()
                if mes_i > 0 and mes_i < minYsizeB1:
                    j += 1
                    continue  # only long cluster  (5*8)
            if inner == 1 and outer > 3:  # B2 and F1
                var mes_i = mes.cast[Int]()
                if mes_i > 0 and mes_i < minYsizeB2:
                    j += 1
                    continue

        var mep = hh.iphi(i.cast[Int]())
        var mer = hh.rGlobal(i.cast[Int]())

        # all cuts: true if fails
        comptime z0cut: Float32 = 12.0  # cm
        comptime hardPtCut: Float32 = 0.5  # GeV
        comptime minRadius: Float32 = hardPtCut * 87.78  # cm
        comptime minRadius2T4: Float32 = 4.0 * minRadius * minRadius

        fn ptcut(j: Int, idphi: Int) -> Bool:
            var r2t4 = minRadius2T4
            var ri = mer
            var ro = hh.rGlobal(j)
            var dphi = ApproxAtan2.short2phi(idphi.cast[Int16]())
            return dphi * dphi * (r2t4 - ri * ro) > (ro - ri) * (ro - ri)

        fn z0cutoff(j: Int) -> Bool:
            var zo = hh.zGlobal(j)
            var ro = hh.rGlobal(j)
            var dr = ro - mer
            return dr > maxr[pairLayerId.cast[Int]()] or dr < 0.0 or abs(mez * ro - mer * zo) > z0cut * dr

        fn zsizeCut(j: Int) -> Bool:
            var onlyBarrel = outer < 4
            var so = hh.clusterSizeY(j)
            var dy = maxDYsize12 if inner == 0 else maxDYsize
            # in the barrel cut on difference in size
            # in the endcap on the prediction on the first layer (actually in the barrel only: happen to be safe for endcap as well)
            # FIXME move pred cut to z0cutoff to optmize loading of and computaiton ...
            var zo = hh.zGlobal(j)
            var ro = hh.rGlobal(j)
            if onlyBarrel:
                var mes_i = mes.cast[Int]()
                var so_i = so.cast[Int]()
                return mes_i > 0 and so_i > 0 and abs(so_i - mes_i) > dy
            var mes_i = mes.cast[Int]()
            return inner < 4 and mes_i > 0 and abs(mes_i -  Int((abs((mez - zo) / (mer - ro)) * dzdrFact + 0.5))) > maxDYPred


        var iphicut = phicuts[pairLayerId.cast[Int]()]
        var kl = Hist.bin((mep - iphicut).cast[Int16]()).cast[UInt32]()
        var kh = Hist.bin((mep + iphicut).cast[Int16]()).cast[UInt32]()
        fn incr(mut k: UInt32) -> UInt32:
            k = (k + 1) % Hist.nbins()
            return k

        @parameter
        if is_defined["GPU_DEBUG"]():
            var tot: Int = 0
            var nmin: Int = 0
            var tooMany: Int = 0

        var kk = kl
        var khh = kh
        _ = incr(khh)
        while kk != khh:
            @parameter
            if is_defined["GPU_DEBUG"]():
                if kk != kl and kk != kh:
                    nmin += hist.size(kk + hoff).cast[Int]()
            var p = hist.begin(kk + hoff)
            var e = hist.end(kk + hoff)
            p += first.cast[Int]()
            while p < e:
                var oi = p[]
                var oi_i = oi.cast[Int]()
                var oi_u = oi.cast[UInt32]()
                assert(oi_u >= offsets[outer_idx])
                assert(oi_u < offsets[outer_idx + 1])
                var mo = hh.detectorIndex(oi_i)
                if mo > 2000:
                    p += stride
                    continue  # invalid

                if doZ0Cut and z0cutoff(oi_i):
                        p += stride
                        continue

                var mop = hh.iphi(oi_i)

                var idphi = min(abs((mop - mep).cast[Int]()), abs((mep - mop).cast[Int]()))
                if idphi > iphicut.cast[Int]():
                    p += stride
                    continue

                if doClusterCut and zsizeCut(oi_i):
                        p += stride
                        continue

                if doPtCut and ptcut(oi_i, idphi):
                        p += stride
                        continue

                var ind = CUDACompat.atomicAdd(nCells, UInt32(1))
                if ind >= maxNumOfDoublets:
                    _ = CUDACompat.atomicSub(nCells, UInt32(1))
                    break
                cells[ind.cast[Int]()].init(
                    cellNeighbors[],
                    cellTracks[],
                    hh,
                    pairLayerId.cast[Int32](),
                    ind.cast[Int32](),
                    i.cast[GPUCACell.hindex_type](),
                    oi.cast[GPUCACell.hindex_type](),
                )
                isOuterHitOfCell[oi_i].push_back(ind)
                @parameter
                if is_defined["GPU_DEBUG"]():
                    if isOuterHitOfCell[oi_i].full():
                        tooMany += 1
                    tot += 1
                p += stride

            _ = incr(kk)

        @parameter
        if is_defined["GPU_DEBUG"]():
            if tooMany > 0:
                print(
                    "OuterHitOfCell full for ",
                    i,
                    " in layer ",
                    inner,
                    "/",
                    outer,
                    ", ",
                    nmin,
                    ",",
                    tot,
                    " ",
                    tooMany,
                )

        j += 1
