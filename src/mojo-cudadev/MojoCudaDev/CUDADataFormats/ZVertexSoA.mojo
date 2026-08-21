# Mojo port of CUDADataFormats/ZVertexSoA.h.
from MojoCudaDev.MojoBridge.DTypes import Float, Typeable


# SoA for vertices, clusterized and fitted only along the beam line (z).
# Global coordinates need the beam spot position added.
struct ZVertexSoA(Movable, Typeable):
    comptime MAXTRACKS: Int = 32 * 1024
    comptime MAXVTX: Int = 1024

    var idv: InlineArray[Int16, Self.MAXTRACKS]  # vertex index per track, -1 if unassociated
    var zv: InlineArray[Float, Self.MAXVTX]  # z-position of found vertices
    var wv: InlineArray[Float, Self.MAXVTX]  # weight (1/error^2) on the above
    var chi2: InlineArray[Float, Self.MAXVTX]
    var ptv2: InlineArray[Float, Self.MAXVTX]
    var ndof: InlineArray[Int32, Self.MAXTRACKS]  # reused as workspace for the neighbour count
    var sortInd: InlineArray[UInt16, Self.MAXVTX]  # sorted index (by pt2), ascending
    var nvFinal: UInt32

    fn __init__(out self):
        self.idv = InlineArray[Int16, Self.MAXTRACKS](fill=0)
        self.zv = InlineArray[Float, Self.MAXVTX](fill=0.0)
        self.wv = InlineArray[Float, Self.MAXVTX](fill=0.0)
        self.chi2 = InlineArray[Float, Self.MAXVTX](fill=0.0)
        self.ptv2 = InlineArray[Float, Self.MAXVTX](fill=0.0)
        self.ndof = InlineArray[Int32, Self.MAXTRACKS](fill=0)
        self.sortInd = InlineArray[UInt16, Self.MAXVTX](fill=0)
        self.nvFinal = 0

    fn init(mut self):
        self.nvFinal = 0

    @staticmethod
    fn dtype() -> String:
        return "ZVertexSoA"
