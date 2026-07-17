import CAConstants
import GPUCACell
import gpuPixelDoubleAlgo
from MojoSerial.CUDADataFormats.TrackingRecHit2DSOAView import TrackingRecHit2DSOAView



alias nPairs: Int = 13 + 2 + 4
@assert(nPairs <= CAConstants.maxNumberOfLayerPairs().cast[Int]())

comptime layerPairs = InlineArray[UInt8, 2 * nPairs](
    0, 1, 0, 4, 0, 7,              # BPIX1 (3)
    1, 2, 1, 4, 1, 7,              # BPIX2 (5)
    4, 5, 7, 8,                    # FPIX1 (8)
    2, 3, 2, 4, 2, 7, 5, 6, 8, 9,  # BPIX3 & FPIX2 (13)
    0, 2, 1, 3,                    # Jumping Barrel (15)
    0, 5, 0, 8,                    # Jumping Forward (BPIX1,FPIX2)
    4, 6, 7, 9                     # Jumping Forward (19)
)

comptime phi0p05: Int16 = 522
comptime phi0p06: Int16 = 626
comptime phi0p07: Int16 = 730

comptime phicuts = InlineArray[Int16, nPairs](
    phi0p05,
    phi0p07,
    phi0p07,
    phi0p05,
    phi0p06,
    phi0p06,
    phi0p05,
    phi0p05,
    phi0p06,
    phi0p06,
    phi0p06,
    phi0p05,
    phi0p05,
    phi0p05,
    phi0p05,
    phi0p05,
    phi0p05,
    phi0p05,
    phi0p05,
)

comptime minz = InlineArray[Float32, nPairs](
    -20.0,
    0.0,
    -30.0,
    -22.0,
    10.0,
    -30.0,
    -70.0,
    -70.0,
    -22.0,
    15.0,
    -30.0,
    -70.0,
    -70.0,
    -20.0,
    -22.0,
    0.0,
    -30.0,
    -70.0,
    -70.0,
)

comptime maxz = InlineArray[Float32, nPairs](
    20.0,
    30.0,
    0.0,
    22.0,
    30.0,
    -10.0,
    70.0,
    70.0,
    22.0,
    30.0,
    -15.0,
    70.0,
    70.0,
    20.0,
    22.0,
    30.0,
    0.0,
    70.0,
    70.0,
)

comptime maxr = InlineArray[Float32, nPairs](
    20.0,
    9.0,
    9.0,
    20.0,
    7.0,
    7.0,
    5.0,
    5.0,
    20.0,
    6.0,
    6.0,
    5.0,
    5.0,
    20.0,
    20.0,
    9.0,
    9.0,
    9.0,
    9.0,
)

alias CellNeighbors = CAConstants.CellNeighbors
alias CellTracks = CAConstants.CellTracks
alias CellNeighborsVector = CAConstants.CellNeighborsVector
alias CellTracksVector = CAConstants.CellTracksVector

fn initDoublets(
    isOuterHitOfCell: UnsafePointer[GPUCACell.OuterHitOfCell],
    nHits: UInt32,
    cellNeighbors: UnsafePointer[CellNeighborsVector],
    cellNeighborsContainer: UnsafePointer[CellNeighbors],
    cellTracks: UnsafePointer[CellTracksVector],
    cellTracksContainer: UnsafePointer[CellTracks],
):
    assert(isOuterHitOfCell)
    var first: UInt32 = 0
    for i in range(first, nHits):
        isOuterHitOfCell[i.cast[Int]()].reset()

    if first == 0:
        cellNeighbors[].construct(
            CAConstants.maxNumOfActiveDoublets().cast[Int32](),
            cellNeighborsContainer,
        )
        cellTracks[].construct(
            CAConstants.maxNumOfActiveDoublets().cast[Int32](),
            cellTracksContainer,
        )
        var i = cellNeighbors[].extend()
        assert(i == 0)
        cellNeighbors[][0].reset()
        i = cellTracks[].extend()
        assert(i == 0)
        cellTracks[][0].reset()


comptime getDoubletsFromHistoMaxBlockSize: Int = 64
comptime getDoubletsFromHistoMinBlocksPerMP: Int = 16
#TO-DO port this 
##ifdef __CUDACC__
# __launch_bounds__(getDoubletsFromHistoMaxBlockSize, getDoubletsFromHistoMinBlocksPerMP)
##endif
fn getDoubletsFromHisto(
    cells: UnsafePointer[GPUCACell],
    nCells: UnsafePointer[UInt32],
    cellNeighbors: UnsafePointer[CellNeighborsVector],
    cellTracks: UnsafePointer[CellTracksVector],
    hhp: UnsafePointer[TrackingRecHit2DSOAView],
    isOuterHitOfCell: UnsafePointer[GPUCACell.OuterHitOfCell],
    nActualPairs: Int,
    ideal_cond: Bool,
    doClusterCut: Bool,
    doZ0Cut: Bool,
    doPtCut: Bool,
    maxNumOfDoublets: UInt32,
):
    var hh = hhp[]
    gpuPixelDoubleAlgo.doubletsFromHisto(
        layerPairs.unsafe_ptr(),
        nActualPairs.cast[UInt32](),
        cells,
        nCells,
        cellNeighbors,
        cellTracks,
        hh,
        isOuterHitOfCell,
        phicuts.unsafe_ptr(),
        minz.unsafe_ptr(),
        maxz.unsafe_ptr(),
        maxr.unsafe_ptr(),
        ideal_cond,
        doClusterCut,
        doZ0Cut,
        doPtCut,
        maxNumOfDoublets,
    )
    
