#
#Original Author: Felice Pantaleo, CERN
#
#
##define NTUPLE_DEBUG



import math

import CAConstants
from MojoSerial.CUDACore.AtomicPairCounter import AtomicPairCounter
from os.atomic import Atomic, Consistency
from memory.unsafe_pointer import UnsafePointer
import gpuPixelDoublets
import GPUCACell
from MojoSerial.CUDACore.CUDACompat import CUDACompat
from CAHitNtupletGeneratorKernels import CAHitNtupletGeneratorKernelsCPU
from MojoSerial.CUDADataFormats.PixelTrackHeterogeneous import (
    PixelTrack as pixelTrack,
    TrackQuality as trackQuality,
)
from MojoSerial.CUDADataFormats.TrackingRecHit2DSOAView import (
    TrackingRecHit2DSOAView,
)
from sys import is_defined

alias HitToTuple = CAConstants.HitToTuple
alias TupleMultiplicity = CAConstants.TupleMultiplicity

alias Quality = pixelTrack.Quality
alias TkSoA = pixelTrack.TrackSoA
alias HitContainer = pixelTrack.HitContainer

fn Kernel_checkOverflows(
    foundNtuplets: UnsafePointer[HitContainer],
    tupleMultiplicity: UnsafePointer[TupleMultiplicity],
    apc: UnsafePointer[AtomicPairCounter],
    cells: UnsafePointer[GPUCACell],  # __restrict__ dropped
    nCells: UnsafePointer[UInt32],    # uint32_t const*
    cellNeighbors: UnsafePointer[gpuPixelDoublets.CellNeighborsVector],
    cellTracks: UnsafePointer[gpuPixelDoublets.CellTracksVector],
    isOuterHitOfCell: UnsafePointer[GPUCACell.OuterHitOfCell],
    nHits: UInt32,
    maxNumberOfDoublets: UInt32,
    counters: UnsafePointer[CAHitNtupletGeneratorKernelsCPU.Counters]
    ):


    var first: UInt32 = 0
    
    ref c = counters[]
    if first == 0 :
        Atomic.fetch_add[ordering = Consistency.SEQUENTIAL](
            UnsafePointer(to=c.nEvents),
            UInt64(1),
        )
        Atomic.fetch_add[ordering = Consistency.SEQUENTIAL](
            UnsafePointer(to=c.nHits),
            nHits.cast[UInt64](),
        )
        Atomic.fetch_add[ordering = Consistency.SEQUENTIAL](
            UnsafePointer(to=c.nCells),
            nCells[].cast[UInt64](),
        )
        Atomic.fetch_add[ordering = Consistency.SEQUENTIAL](
            UnsafePointer(to=c.nTuples),
            apc[].get()[1].cast[UInt64](),
        )
        Atomic.fetch_add[ordering = Consistency.SEQUENTIAL](
            UnsafePointer(to=c.nFitTracks),
            tupleMultiplicity[].size().cast[UInt64](),
        )

    @parameter
    if is_defined["NTUPLE_DEBUG"]():
        if first == 0:
            print(
                "number of found cells",
                nCells[],
                "found tuples",
                apc[].get()[1],
                "with total hits",
                apc[].get()[0],
                "out of",
                nHits,
            )
            if apc[].get()[1] < CAConstants.maxNumberOfQuadruplets():
                debug_assert(
                    foundNtuplets[].size(apc[].get()[1]) == 0,
                    "Expected size 0",
                )
                debug_assert(
                    foundNtuplets[].size() == apc[].get()[0],
                    "Size mismatch",
                )
    @parameter
    if is_defined["NTUPLE_DEBUG"]():
        var nBins = foundNtuplets[].nbins().cast[Int]()
        for idx in range(first.cast[Int](), nBins, 1):
            var idx_u = idx.cast[UInt32]()
            if foundNtuplets[].size(idx_u) > 5:
                print("ERROR", idx, ",", foundNtuplets[].size(idx_u))
            debug_assert(foundNtuplets[].size(idx_u) < 6)

            var ih = foundNtuplets[].begin(idx_u)
            var end = foundNtuplets[].end(idx_u)
            while ih != end:
                debug_assert(ih[] < nHits)
                ih += 1
    if first == 0:
        if apc[].get()[1] >= CAConstants.maxNumberOfQuadruplets():
            print("Tuples overflow")
        if nCells[] >= maxNumberOfDoublets:
            print("Cells overflow")
        if cellNeighbors and cellNeighbors[].full():
            print("cellNeighbors overflow")
        if cellTracks and cellTracks[].full():
            print("cellTracks overflow")

    var idx: Int = first.cast[Int]()
    var nt = nCells[].cast[Int]()

    while idx  < nt:
        ref thisCell = (cells + idx)[]
        if (thisCell.outerNeighbors().full()) : #++tooManyNeighbors[thisCell.theLayerPairId]
          print("OuterNeighbors overflow ",idx , "in \n", thisCell.theLayerPairId)
        if (thisCell.tracks().full()) : #++tooManyTracks[thisCell.theLayerPairId]
          print("Tracks overflow " , idx , " in \n", thisCell.theLayerPairId)
        if (thisCell.theDoubletId < 0):
          Atomic.fetch_add[ordering = Consistency.SEQUENTIAL](
              UnsafePointer(to=c.nKilledCells),
              UInt64(1),
          )
        if (0 == thisCell.theUsed):
          Atomic.fetch_add[ordering = Consistency.SEQUENTIAL](
              UnsafePointer(to=c.nEmptyCells),
              UInt64(1),
          )
        if (thisCell.tracks().empty()):
          Atomic.fetch_add[ordering = Consistency.SEQUENTIAL](
              UnsafePointer(to=c.nZeroTrackCells),
              UInt64(1),
          )
        idx+=1
        
  
    idx = first.cast[Int]()
    nt = nHits.cast[Int]()

    while idx < nt :
        if isOuterHitOfCell[idx].full():
            print("OuterHitOfCell ovberflow " , idx , "\n")

        idx += 1
        

fn kernel_fishboneCleaner(
    cells: UnsafePointer[GPUCACell],
    nCells: UnsafePointer[UInt32],
    quality: UnsafePointer[Quality],
):
    comptime bad = trackQuality.bad
    var nt = nCells[].cast[Int]()
    for idx in range(0, nt, 1):
        ref thisCell = (cells + idx)[]
        if thisCell.theDoubletId >= 0:
            continue

        for it in thisCell.tracks():
            quality[it.cast[Int]()] = bad
        

fn kernel_earlyDuplicateRemover(
    cells: UnsafePointer[GPUCACell],
    nCells: UnsafePointer[UInt32],
    foundNtuplets: UnsafePointer[HitContainer],
    quality: UnsafePointer[Quality]
    ):

    comptime dup = trackQuality.dup

    assert nCells != 0
    var nt = nCells[].cast[Int]()
    for idx in range(0, nt, 1):
        ref thisCell = (cells + idx)[]

        if len(thisCell.tracks()) < 2:
            continue

        var maxNh: UInt32 = 0

        for it in thisCell.tracks():
            var nh = foundNtuplets[].size(it.cast[UInt32]())
            maxNh = math.max(nh, maxNh)

        for it in thisCell.tracks():
            if foundNtuplets[].size(it.cast[UInt32]()) != maxNh:
                quality[it.cast[Int]()] = dup
               

fn kernel_fastDuplicateRemover(
    cells: UnsafePointer[GPUCACell], 
    nCells: UnsafePointer[UInt32],
    foundNtuplets: UnsafePointer[HitContainer],
    tracks: UnsafePointer[TkSoA]
) :
    var bad = trackQuality.bad
    var dup = trackQuality.dup
    var loose = trackQuality.loose

    assert nCells != 0

    var nt = nCells[].cast[Int]()
    for idx in range(0, nt, 1):
        ref thisCell = (cells + idx)[]
        if len(thisCell.tracks()) < 2:
            continue

        var mc: Float32 = 10000.0
        var im: UInt16 = 60000

        fn score(it) -> Float32:
            return math.abs(tracks[].tip(it.cast[Int32]())) # tip
            # or chi2
        #find min socre
        for it in thisCell.tracks():
            if tracks[].quality(it.cast[Int]()) == loose and score(it) < mc:
                mc = score(it)
                im = it 
        #mark all other duplicates
        for it in thisCell.tracks():
            if tracks[].quality(it.cast[Int]()) != bad and it != im:
                tracks[].quality(it.cast[Int]()) = dup # no race:  simple assignment of the same constant
        
                

fn kernel_connect(
            apc1: UnsafePointer[AtomicPairCounter],
            apc2: UnsafePointer[AtomicPairCounter],  # just to zero them
            hhp: UnsafePointer[GPUCACell.Hits],              
            cells: UnsafePointer[GPUCACell],
            nCells: UnsafePointer[UInt32],
            cellNeighbors: UnsafePointer[gpuPixelDoublets.CellNeighborsVector],
            isOuterHitOfCell: UnsafePointer[GPUCACell.OuterHitOfCell],
            hardCurvCut: Float32,
            ptmin: Float32,
            CAThetaCutBarrel: Float32,
            CAThetaCutForward: Float32,
            dcaCutInnerTriplet: Float32,
            dcaCutOuterTriplet: Float32
        ):
    ref hh = hhp[]
    
    var firstCellIndex = 0 + 0 * 1
    var first: UInt32 = 0
    var stride = 1

    if(0 == (firstCellIndex + first.cast[Int]())):
        apc1[] = 0
        apc2[] = 0

    var idx : Int = firstCellIndex 
    var nt = nCells[].cast[Int]()
    #loop on outer cells
    while idx < nt:
        var cellIndex  =idx
        ref thisCell  = (cells + idx)[]

        var innerHitId = thisCell.get_inner_hit_id()
        var innerHitIdx = innerHitId.cast[Int]()
        var numberOfPossibleNeighbors : Int = len(isOuterHitOfCell[innerHitIdx])
        var vi = isOuterHitOfCell[innerHitIdx].data()

        var last_bpix1_detIndex: UInt32 = 96
        var last_barrel_detIndex: UInt32 = 1184
        var ri = thisCell.get_inner_r(hh)
        var zi = thisCell.get_inner_z(hh)

        var ro = thisCell.get_outer_r(hh)
        var zo = thisCell.get_outer_z(hh)
        var isBarrel = thisCell.get_inner_detIndex(hh) < last_barrel_detIndex.cast[Float32]()
        #loop on inner cells
        for j in range(first.cast[Int](), numberOfPossibleNeighbors, stride):
            var otherCell = vi[j]
            ref oc = (cells + otherCell.cast[Int]())[]

            var r1 = oc.get_inner_r(hh)
            var z1 = oc.get_inner_z(hh)
            
            var aligned : Bool = GPUCACell.areAlignedRZ(
                r1,
                z1,
                ri,
                zi,
                ro,
                zo,
                ptmin,
                CAThetaCutBarrel if isBarrel else  CAThetaCutForward
            )
            if aligned and thisCell.dcaCut(
                hh,
                oc,
                dcaCutInnerTriplet
                if oc.get_inner_detIndex(hh) < last_bpix1_detIndex.cast[Float32]()
                else dcaCutOuterTriplet,
                hardCurvCut,
            ):
                oc.addOuterNeighbor(cellIndex.cast[UInt32](), cellNeighbors[])
                thisCell.theUsed |= 1
                oc.theUsed |= 1
        idx += 1

fn kernel_find_ntuplets(
    hhp : UnsafePointer[GPUCACell.Hits],
    cells : UnsafePointer[GPUCACell],
    nCells : UnsafePointer[UInt32],
    cellTracks : UnsafePointer[gpuPixelDoublets.CellTracksVector],
    foundNtuplets : UnsafePointer[HitContainer],
    apc : UnsafePointer[AtomicPairCounter],
    quality : UnsafePointer[Quality],
    minHitsPerNtuplet : UInt32
    ):

    ref hh = hhp[]

    var nt = nCells[].cast[Int]()
    for idx in range(0, nt, 1):
        ref thisCell =  (cells + idx)[]
        if thisCell.theDoubletId < 0:
            continue

        var pid = thisCell.theLayerPairId.cast[Int]()
        var doit: Bool = (pid < 3) if minHitsPerNtuplet > 3 else (pid < 8 or pid > 12)
        if doit:
            var stack = GPUCACell.TmpTuple()
            stack.reset()
            thisCell.find_ntuplets[6](
                hh,
                cells,
                cellTracks[],
                foundNtuplets[],
                apc[],
                quality,
                stack,
                minHitsPerNtuplet,
                pid < 3,
            )
            assert stack.empty()

fn kernel_mark_used(
    hhp: UnsafePointer[GPUCACell.Hits],
    cells: UnsafePointer[GPUCACell],
    nCells: UnsafePointer[UInt32],
):
    var nt = nCells[].cast[Int]()
    for idx in range(0, nt, 1):
        ref thisCell = (cells + idx)[]
        if not thisCell.tracks().empty():
            thisCell.theUsed |= 2

fn kernel_countMultiplicity(  foundNtuplets : UnsafePointer[HitContainer],
                              quality : UnsafePointer[Quality],
                              tupleMultiplicity : UnsafePointer[CAConstants.TupleMultiplicity ]):
    var nt = foundNtuplets[].nbins().cast[Int]()
    for it in range(0, nt, 1):
        var it_u = it.cast[UInt32]()
        var nhits = foundNtuplets[].size(it_u)
        if nhits < 3:
            continue
        if quality[it] == trackQuality.dup:
            continue
        assert quality[it] == trackQuality.bad
        if nhits > 5:
            print("wrong mult", it, nhits)
        assert nhits < 8
        tupleMultiplicity[].countDirect(nhits)

fn kernel_fillMultiplicity(foundNtuplets : UnsafePointer[HitContainer],
                              quality : UnsafePointer[Quality],
                              tupleMultiplicity : UnsafePointer[CAConstants.TupleMultiplicity ]):
    var nt = foundNtuplets[].nbins().cast[Int]()
    for it in range(0, nt, 1):
        var it_u = it.cast[UInt32]()
        var nhits = foundNtuplets[].size(it_u)
        if nhits < 3:
            continue
        if quality[it] == trackQuality.dup:
            continue
        assert quality[it] == trackQuality.bad 
        if nhits > 5:
            print("wrong mult", it, nhits)
        assert nhits < 8
        tupleMultiplicity[].fillDirect(
            nhits,
            it_u.cast[CAConstants.tindex_type](),
        )

fn kernel_classifyTracks(
    tuples: UnsafePointer[HitContainer],
    tracks: UnsafePointer[TkSoA],
    cuts: CAHitNtupletGeneratorKernelsCPU.QualityCuts,
    quality: UnsafePointer[Quality],
):
    var nt = tuples[].nbins().cast[Int]()
    for it in range(0, nt, 1):
        var it_u = it.cast[UInt32]()
        var it_i = it.cast[Int32]()
        var nhits = tuples[].size(it_u)
        if nhits == 0:
            break # guard
        
        #id duplicate : not even fit 
        if quality[it] == trackQuality.dup:
            continue

        assert quality[it] == trackQuality.bad 

        #mark doublets as bad 
        if nhits < 3:
            continue

        #if the fit has my invalid parameters , mark it as bad
        var isNaN : Bool = false 
        for i in range(0, 5, 1):
            isNaN = isNaN or math.isnan(tracks[].stateAtBS.state[it_i][i, 0])

        if isNaN:
            @parameter
            if is_defined["NTUPLE_DEBUG"]():
                print(
                    "NaN in fit",
                    it,
                    "size",
                    nhits,
                    "chi2",
                    tracks[].chi2[it],
                )
            continue
        # compute a pT-dependent chi2 cut
        # default parameters:
        #   - chi2MaxPt = 10 GeV
        #   - chi2Coeff = { 0.68177776, 0.74609577, -0.08035491, 0.00315399 }
        #   - chi2Scale = 30 for broken line fit, 45 for Riemann fit
        # (see CAHitNtupletGeneratorGPU.cc)
        var pt :float32= min(tracks[].pt[it], cuts.chi2MaxPt)
        var chi2Cut: Float32 = cuts.chi2Scale * (
            cuts.chi2Coeff[0]
            + pt
            * (cuts.chi2Coeff[1] + pt * (cuts.chi2Coeff[2] + pt * cuts.chi2Coeff[3]))
        )
        # above number were for Quads not normalized so for the time being just multiple by ndof for Quads  (triplets to be understood)
        if 3.0 * tracks[].chi2[it] >= chi2Cut:
            @parameter
            if is_defined["NTUPLE_DEBUG"]():
                print(
                    "Bad fit",
                    it,
                    "size",
                    nhits,
                    "pt",
                    tracks[].pt[it],
                    "eta",
                    tracks[].eta[it],
                    "chi2",
                    3.0 * tracks[].chi2[it],
                )
            continue
        # impose "region cuts" based on the fit results (phi, Tip, pt, cotan(theta)), Zip)
        # default cuts:
        #   - for triplets:    |Tip| < 0.3 cm, pT > 0.5 GeV, |Zip| < 12.0 cm
        #   - for quadruplets: |Tip| < 0.5 cm, pT > 0.3 GeV, |Zip| < 12.0 cm
        # (see CAHitNtupletGeneratorGPU.cc)  
        var region = cuts.quadruplet if nhits > 3 else cuts.triplet
        var tip = tracks[].tip(it_i)
        var zip = tracks[].zip(it_i)
        var isOk: Bool = (
            math.abs(tip) < region.maxTip
            and tracks[].pt[it] > region.minPt
            and math.abs(zip) < region.maxZip
        )

        if isOk:
            quality[it] = trackQuality.loose

fn kernel_doStatsForTracks(tuples  : UnsafePointer[HitContainer] , 
                           quality : UnsafePointer[Quality] , 
                           counters : UnsafePointer[CAHitNtupletGeneratorKernelsCPU.Counters] ):
    var nt = tuples[].nbins().cast[Int]()
    for idx in range(0, nt, 1):
        var idx_u = idx.cast[UInt32]()
        if tuples[].size(idx_u) == 0:
            break # guard
        if quality[idx] != trackQuality.loose:
            continue
        Atomic.fetch_add[ordering = Consistency.SEQUENTIAL](
            UnsafePointer(to=(counters[].nGoodTracks)),
            UInt64(1),
        )


fn kernel_countHitInTracks(tuples  : UnsafePointer[HitContainer] , 
                           quality : UnsafePointer[Quality] , 
                           hitToTuple : UnsafePointer[CAHitNtupletGeneratorKernelsCPU.HitToTuple] ):
    var nt = tuples[].nbins().cast[Int]()
    for idx in range(0, nt, 1):
        var idx_u = idx.cast[UInt32]()
        if tuples[].size(idx_u) == 0:
            break # guard
        if quality[idx] != trackQuality.loose:
            continue
        var h  = tuples[].begin(idx_u)
        var end = tuples[].end(idx_u)
        while h != end:
            hitToTuple[].countDirect(h[].cast[UInt32]())
            h += 1

fn kernel_fillHitInTracks(tuples  : UnsafePointer[HitContainer] , 
                           quality : UnsafePointer[Quality] , 
                           hitToTuple : UnsafePointer[CAHitNtupletGeneratorKernelsCPU.HitToTuple] ):
    var nt = tuples[].nbins().cast[Int]()

    for idx in range(0, nt, 1):
        var idx_u = idx.cast[UInt32]()
        if tuples[].size(idx_u) == 0:
            break #guard
        if quality[idx] != trackQuality.loose:
            continue
        var h  = tuples[].begin(idx_u)
        var end = tuples[].end(idx_u)
        while h != end:
            hitToTuple[].fillDirect(
                h[].cast[UInt32](),
                idx_u.cast[CAConstants.tindex_type](),
            )
            h += 1

fn kernel_fillHitDetIndices(tuples  : UnsafePointer[HitContainer] , 
                           hhp : UnsafePointer[TrackingRecHit2DSOAView] , 
                           hitDetIndices : UnsafePointer[HitContainer] ):
    # copy offsets
    var total_bins = tuples[].totbins().cast[Int]()
    for idx in range(0, total_bins, 1):
        hitDetIndices[].off[idx] = tuples[].off[idx]
    # fill hit indices
    ref hh = hhp[]
    var nhits = hh.nHits()
    var total_size = tuples[].size().cast[Int]()
    for idx in range(0, total_size, 1):
        assert tuples[].bins[idx] < nhits
        hitDetIndices[].bins[idx] = hh.detectorIndex(tuples[].bins[idx])

fn kernel_doStatsForHitInTracks(hitToTuple: UnsafePointer[CAHitNtupletGeneratorKernelsCPU.HitToTuple] ,counters :  UnsafePointer[CAHitNtupletGeneratorKernelsCPU.Counters]):
    ref c = counters[]
    var nt = hitToTuple[].nbins().cast[Int]()
    for idx in range(0, nt, 1):
        var idx_u = idx.cast[UInt32]()
        if hitToTuple[].size(idx_u) == 0:
            continue # SHALL NOT BE break
        Atomic.fetch_add[ordering = Consistency.SEQUENTIAL](
            UnsafePointer(to=c.nUsedHits),
            UInt64(1),
        )
        if hitToTuple[].size(idx) > 1 :
            Atomic.fetch_add[ordering = Consistency.SEQUENTIAL](
                UnsafePointer(to=c.nDupHits),
                UInt64(1),
            )

fn kernel_tripletCleaner(hhp : UnsafePointer[TrackingRecHit2DSOAView] , ptuples : UnsafePointer[HitContainer] , ptracks : UnsafePointer[TkSoA] , quality : UnsafePointer[Quality] , phitToTuple : UnsafePointer[CAHitNtupletGeneratorKernelsCPU.HitToTuple]):
    var bad  = trackQuality.bad
    var dup  = trackQuality.dup

    ref hitToTuple = phitToTuple[]
    ref foundNtuplets = ptuples[]
    ref tracks = ptracks[]

    # loop over hits
    var nt = hitToTuple.nbins().cast[Int]()
    for idx in range(0, nt, 1):
        var idx_u = idx.cast[UInt32]()
        if hitToTuple.size(idx_u) < 2:
            continue
        
        var mc: Float32 = 10000.0
        var im: UInt16 = 60000
        var maxNh: UInt32 = 0

        # find maxNh 
        var it = hitToTuple.begin(idx_u)
        var it_end = hitToTuple.end(idx_u)
        while it != it_end:
            var nh: UInt32 = foundNtuplets.size(it[].cast[UInt32]())
            maxNh = math.max(nh, maxNh)
            it += 1
        
        # kill all tracks shorter than maxHn (only triplets???)
        it = hitToTuple.begin(idx_u)
        while it != it_end:
            var nh: UInt32 = foundNtuplets.size(it[].cast[UInt32]())
            if maxNh != nh:
                quality[it[].cast[Int]()] = dup
            it += 1

        if maxNh > 3:
            continue

        # for triplets choose best tip!
        var ip = hitToTuple.begin(idx_u)
        var ip_end = hitToTuple.end(idx_u)
        while ip != ip_end:
            var it_val = ip[]
            var it_i = it_val.cast[Int32]()
            if quality[it_val.cast[Int]()] != bad and math.abs(tracks.tip(it_i)) < mc:
                mc = math.abs(tracks.tip(it_i))
                im = it_val
            ip += 1

        ip = hitToTuple.begin(idx_u)
        while ip != ip_end:
            var it_val = ip[]
            if quality[it_val.cast[Int]()] != bad and it_val != im:
                quality[it_val.cast[Int]()] = dup # no race:  simple assignment of the same constant
            ip += 1


fn kernel_print_found_ntuplets(hhp : UnsafePointer[TrackingRecHit2DSOAView] , ptuples : UnsafePointer[HitContainer] , ptracks : UnsafePointer[TkSoA] , quality : UnsafePointer[Quality] , phitToTuple : UnsafePointer[CAHitNtupletGeneratorKernelsCPU.HitToTuple] ,  maxPrint : UInt32  ,  iev : Int):
    ref foundNtuplets = ptuples[]
    ref tracks = ptracks[]
   

    var i: Int = 0
    while i < min(maxPrint, foundNtuplets.nbins()).cast[Int]():
        var i_u = i.cast[UInt32]()
        var nh = foundNtuplets.size(i_u)
        if nh < 3: 
            i += 1
            continue
        print(
            "TK:",
            10000 * iev + i,
            Int(quality[i]),
            nh,
            tracks.charge(i.cast[Int32]()),
            tracks.pt[i],
            tracks.eta[i],
            tracks.phi(i.cast[Int32]()),
            tracks.tip(i.cast[Int32]()),
            tracks.zip(i.cast[Int32]()),
            tracks.chi2[i],
            (foundNtuplets.begin(i_u))[],
            (foundNtuplets.begin(i_u) + 1)[],
            (foundNtuplets.begin(i_u) + 2)[],
            Int((foundNtuplets.begin(i_u) + 3)[]) if nh > 3 else -1,
            Int((foundNtuplets.begin(i_u) + 4)[]) if nh > 4 else -1,
        )
        i += 1

fn kernel_printCounters(
    counters: UnsafePointer[CAHitNtupletGeneratorKernelsCPU.Counters],
):
    ref c = counters[]
    print(
        "||Counters | nEvents | nHits | nCells | nTuples | nFitTracks | nGoodTracks | nUsedHits | nDupHits | "
        "nKilledCells | nEmptyCells | nZeroTrackCells ||"
    )
    print(
        "Counters Raw",
        c.nEvents,
        c.nHits,
        c.nCells,
        c.nTuples,
        c.nGoodTracks,
        c.nFitTracks,
        c.nUsedHits,
        c.nDupHits,
        c.nKilledCells,
        c.nEmptyCells,
        c.nZeroTrackCells,
    )
    var events = c.nEvents.cast[Float64]()
    var cells = c.nCells.cast[Float64]()
    print(
        "Counters Norm",
        c.nEvents,
        c.nHits.cast[Float64]() / events,
        c.nCells.cast[Float64]() / events,
        c.nTuples.cast[Float64]() / events,
        c.nFitTracks.cast[Float64]() / events,
        c.nGoodTracks.cast[Float64]() / events,
        c.nUsedHits.cast[Float64]() / events,
        c.nDupHits.cast[Float64]() / events,
        c.nKilledCells.cast[Float64]() / events,
        c.nEmptyCells.cast[Float64]() / cells,
        c.nZeroTrackCells.cast[Float64]() / cells,
    )
