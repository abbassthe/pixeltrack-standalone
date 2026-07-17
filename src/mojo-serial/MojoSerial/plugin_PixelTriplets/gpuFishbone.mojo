
from memory.unsafe_pointer import UnsafePointer
from builtin.type_aliases import ImmutOrigin, MutOrigin
from builtin.uint import UInt32
from builtin.bool import Bool

trait gpuPixelDoublets:

    
    fn fishbone(
        hhp: UnsafePointer[GPUCACell.Hits, ImmutOrigin.external],          
        cells: UnsafePointer[GPUCACell, MutOrigin.external],             
        nCells: UnsafePointer[UInt32, ImmutOrigin.external],            
        isOuterHitOfCell: UnsafePointer[GPUCACell.OuterHitOfCell, ImmutOrigin.external],  
        nHits: UInt32,                                                    
        checkTrack: Bool                                                  
        ):

        comptime maxCellsPerHit = GPUCACell.maxCellsPerHit

        var hh = hhp[]

        var firstY = 0 + 0 * 1
        var firstX : UInt32 = 0

        var x = InlineArray[Float32, maxCellsPerHit]()
        var y = InlineArray[Float32, maxCellsPerHit]()
        var z = InlineArray[Float32, maxCellsPerHit]()
        var n = InlineArray[Float32, maxCellsPerHit]()
        

        var d = InlineArray[UInt16, maxCellsPerHit]()
        var cc = InlineArray[UInt32 , maxCellsPerHit]()

        var nt = nHits
        for idy in range(firstY, nt):
            ref vc  = isOuterHitOfCell[idy]
            var s = vc.size()

            if s < 2 :
                continue
            var c0 = cells[vc[0]]
            var xo = c0.get_outer_x(hh)
            var yo = c0.get_outer_y(hh)
            var zo = c0.get_outer_z(hh)
            var sg = 0
            for ic in range(s):
                ref ci = cells[vc[ic]]
                if 0 == ci.theUsed:
                    continue
                if checkTrack and ci.tracks().empty():
                    continue
                cc[sg] = vc[ic]
                d[sg] = ci.get_inner_detIndex(hh)

                x[sg] = ci.get_inner_x(hh) - xo
                y[sg] = ci.get_inner_y(hh) - yo
                z[sg] = ci.get_inner_z(hh) - zo
                n[sg] = x[sg] * x[sg] + y[sg] * y[sg] + z[sg] * z[sg]
                sg += 1
        
        if sg < 2 :
            continue
        
        for ic in range(firstX , sg - 1):
            ref ci = cells[cc[ic]]
            for jc in range(ic + 1 , sg):
                cj = cells[cc[jc]]
                var cos12 = x[ic] * x[jc] + y[ic] * y[jc] + z[ic] * z[jc]
                if d[ic] != d[jc] and cos12 * cos12 >= 0.99999 * n[ic] * n[jc]:
                    if n[ic] > n[jc]:
                        ci.theDoubletId = -1
                        break
                    else :
                        cj.theDoubletId = -1