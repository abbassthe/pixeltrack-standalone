from MojoSerial.CUDACore.CUDACompat import CUDACompat
from MojoSerial.MojoBridge.DTypes import Float
from MojoSerial.plugin_PixelVertexFinding.gpuVertexFinder import ZVertices, WorkSpace


@always_inline
def sortByPt2(mut data: ZVertices, mut ws: WorkSpace):
    var nt: UInt32 = ws.ntrks
    ref ptt2 = ws.ptt2
    var nvFinal: UInt32 = data.nvFinal

    ref iv = ws.iv
    ref ptv2 = data.ptv2
    ref sortInd = data.sortInd

    if nvFinal < 1:
        return

    # fill indexing
    for i in range(nt):
        data.idv[Int(ws.itrk[i])] = Int16(iv[i])

    # can be done asynchronously at the end of the previous event
    for i in range(nvFinal):
        ptv2[i] = 0

    for i in range(nt):
        if iv[i] > 9990:
            continue
        _ = CUDACompat.atomicAdd(ptv2[Int(iv[i])], ptt2[i])

    if nvFinal == 1:
        sortInd[0] = 0
        return

    for i in range(nvFinal):
        sortInd[i] = UInt16(i)

    # C++ sorts [sortInd, sortInd + nvFinal); the slice is that same range.
    var sortIndSpan = Span(sortInd)[: Int(nvFinal)]

    @parameter
    def less_than(i: UInt16, j: UInt16) -> Bool:
        return ptv2[Int(i)] < ptv2[Int(j)]

    sort[less_than](sortIndSpan)
