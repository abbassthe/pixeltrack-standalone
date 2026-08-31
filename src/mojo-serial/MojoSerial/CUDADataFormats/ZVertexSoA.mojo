from MojoSerial.MojoBridge.DTypes import Float, Typeable


@fieldwise_init
struct ZVertexSoA(Copyable, Defaultable, Movable, Typeable):
    comptime MAXTRACKS: UInt32 = 32 * 1024
    comptime MAXVTX: UInt32 = 1024

    var idv: InlineArray[
        Int16, UInt(Self.MAXTRACKS)
    ]  # vertex index for each associated (original) track  (-1 == not associate)
    var zv: InlineArray[
        Float, UInt(Self.MAXVTX)
    ]  # output z-position of found vertices
    var wv: InlineArray[
        Float, UInt(Self.MAXVTX)
    ]  # output weight (1/error^2) on the above
    var chi2: InlineArray[Float, UInt(Self.MAXVTX)]  # vertices chi2
    var ptv2: InlineArray[Float, UInt(Self.MAXVTX)]  # vertices pt^2
    var ndof: InlineArray[Int32, UInt(Self.MAXTRACKS)]  # vertices number of dof
    var sortInd: InlineArray[
        UInt16, UInt(Self.MAXVTX)
    ]  # sorted index (by pt2)  ascending
    var nvFinal: UInt32  # the number of vertices

    @always_inline
    def __init__(out self):
        self.idv = InlineArray[Int16, UInt(Self.MAXTRACKS)](fill=0)
        self.zv = InlineArray[Float, UInt(Self.MAXVTX)](fill=0.0)
        self.wv = InlineArray[Float, UInt(Self.MAXVTX)](fill=0.0)
        self.chi2 = InlineArray[Float, UInt(Self.MAXVTX)](fill=0.0)
        self.ptv2 = InlineArray[Float, UInt(Self.MAXVTX)](fill=0.0)
        self.ndof = InlineArray[Int32, UInt(Self.MAXTRACKS)](fill=0)
        self.sortInd = InlineArray[UInt16, UInt(Self.MAXVTX)](fill=0)
        self.nvFinal = 0

    @always_inline
    def init(mut self):
        self.nvFinal = 0

    @always_inline
    @staticmethod
    def dtype() -> String:
        return "ZVertexSoA"
