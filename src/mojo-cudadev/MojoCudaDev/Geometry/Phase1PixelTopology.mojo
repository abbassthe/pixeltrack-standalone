from MojoCudaDev.MojoBridge.DTypes import Float, Typeable


struct Phase1PixelTopology:
    alias numRowsInRoc: UInt16 = 80
    alias numColsInRoc: UInt16 = 52
    alias lastRowInRoc: UInt16 = Self.numRowsInRoc - 1
    alias lastColInRoc: UInt16 = Self.numColsInRoc - 1

    alias numRowsInModule: UInt16 = 2 * Self.numRowsInRoc
    alias numColsInModule: UInt16 = 8 * Self.numColsInRoc
    alias lastRowInModule: UInt16 = Self.numRowsInModule - 1
    alias lastColInModule: UInt16 = Self.numColsInModule - 1

    alias xOffset: Int16 = -81
    alias yOffset: Int16 = -54 * 4

    alias numPixsInModule: UInt32 = Self.numRowsInModule.cast[
        DType.uint32
    ]() * Self.numColsInModule.cast[DType.uint32]()

    alias numberOfModules: UInt32 = 1856
    alias numberOfLayers: UInt32 = 10
    alias layerStart: InlineArray[UInt32, Int(Self.numberOfLayers) + 1] = [
        0,
        96,
        320,
        672,  # barrel
        1184,
        1296,
        1408,  # positive endcap
        1520,
        1632,
        1744,  # negative endcap
        Self.numberOfModules,
    ]

    alias layerName: InlineArray[StaticString, Int(Self.numberOfLayers)] = [
        "BL1",
        "BL2",
        "BL3",
        "BL4",  # barrel
        "E+1",
        "E+2",
        "E+3",  # positive endcap
        "E-1",
        "E-2",
        "E-3",  # negative endcap
    ]

    alias numberOfModulesInBarrel: UInt32 = 1184
    alias numberOfLaddersInBarrel: UInt32 = Self.numberOfModulesInBarrel / 8

    @staticmethod
    fn _map_to_array[
        I: DType, R: DType, N: Int, func: fn[Scalar[I]] () -> Scalar[R]
    ]() -> InlineArray[Scalar[R], N]:
        var arr = InlineArray[Scalar[R], N](fill=0)

        @parameter
        for i in range(N):
            arr[i] = func[i]()
        return arr

    @staticmethod
    fn findMaxModuleStride() -> UInt32:
        var go = True
        var n = 2
        while go:

            @parameter
            for i in range(1, 11):
                if Self.layerStart[i] % n != 0:
                    go = False
                    break
            if not go:
                break
            n *= 2
        return n // 2

    alias maxModuleStride = Self.findMaxModuleStride()

    @staticmethod
    fn findLayer[detId: UInt32]() -> UInt8:
        @parameter
        for i in range(11):
            if detId < Self.layerStart[i + 1]:
                return i
        return 11

    @staticmethod
    fn findLayerFromCompact[detId: UInt32]() -> UInt8:
        alias _detId = detId * Self.maxModuleStride

        @parameter
        for i in range(11):
            if _detId < Self.layerStart[i + 1]:
                return i
        return 11

    alias layerIndexSize: UInt32 = Self.numberOfModules / Self.maxModuleStride

    alias layer: InlineArray[
        UInt8, Int(Self.layerIndexSize)
    ] = Self._map_to_array[
        DType.uint32,
        DType.uint8,
        Int(Self.layerIndexSize),
        Self.findLayerFromCompact,
    ]()

    @staticmethod
    fn validateLayerIndex() -> Bool:
        var res = True
        for i in range(Self.numberOfModules):
            var j = i / Self.maxModuleStride
            res &= Self.layer[j] < 10
            res &= i >= Self.layerStart[Self.layer[j]]
            res &= i < Self.layerStart[Self.layer[j] + 1]
        return res

    alias __d = debug_assert(
        Self.validateLayerIndex(), "layer from detIndex algo is buggy"
    )

    @always_inline
    @staticmethod
    fn divu52(var n: UInt16) -> UInt16:
        """
        This is for the ROC n<512 (upgrade 1024).
        """
        n = n >> 2
        var q = (n >> 1) + (n >> 4)
        q = q + (q >> 4) + (q >> 5)
        q = q >> 3
        var r = n - q * 13
        return q + ((r + 3) >> 4)

    @staticmethod
    @always_inline
    fn isEdgeX(px: UInt16) -> Bool:
        return px == 0 or px == Self.lastRowInModule

    @staticmethod
    @always_inline
    fn isEdgeY(py: UInt16) -> Bool:
        return py == 0 or py == Self.lastColInModule

    @staticmethod
    @always_inline
    fn toRocX(px: UInt16) -> UInt16:
        return px if px < Self.numRowsInRoc else px - Self.numRowsInRoc

    @staticmethod
    @always_inline
    fn toRocY(py: UInt16) -> UInt16:
        return py - 52 * Self.divu52(py)

    @staticmethod
    @always_inline
    fn isBigPixX(px: UInt16) -> Bool:
        return px == 79 or px == 80

    @staticmethod
    @always_inline
    fn isBigPixY(py: UInt16) -> Bool:
        var ly = Self.toRocY(py)
        return ly == 0 or ly == Self.lastColInRoc

    @staticmethod
    @always_inline
    fn localX(px: UInt16) -> UInt16:
        var shift: UInt16 = 0
        if px > Self.lastRowInRoc:
            shift += 1
        if px > Self.numRowsInRoc:
            shift += 1
        return px + shift

    @staticmethod
    @always_inline
    fn localY(py: UInt16) -> UInt16:
        var roc = Self.divu52(py)
        var shift = 2 * roc
        var yInRoc = py - 52 * roc
        if yInRoc > 0:
            shift += 1
        return py + shift


@fieldwise_init
struct AverageGeometry(Defaultable, Movable, Typeable):
    alias numberOfLaddersInBarrel = Phase1PixelTopology.numberOfLaddersInBarrel
    var ladderZ: InlineArray[Float, Int(Self.numberOfLaddersInBarrel)]
    var ladderX: InlineArray[Float, Int(Self.numberOfLaddersInBarrel)]
    var ladderY: InlineArray[Float, Int(Self.numberOfLaddersInBarrel)]
    var ladderR: InlineArray[Float, Int(Self.numberOfLaddersInBarrel)]
    var ladderMinZ: InlineArray[Float, Int(Self.numberOfLaddersInBarrel)]
    var ladderMaxZ: InlineArray[Float, Int(Self.numberOfLaddersInBarrel)]
    var endCapZ: InlineArray[Float, 2]

    fn __init__(out self):
        self.ladderZ = InlineArray[Float, Int(Self.numberOfLaddersInBarrel)](
            fill=0
        )
        self.ladderX = InlineArray[Float, Int(Self.numberOfLaddersInBarrel)](
            fill=0
        )
        self.ladderY = InlineArray[Float, Int(Self.numberOfLaddersInBarrel)](
            fill=0
        )
        self.ladderR = InlineArray[Float, Int(Self.numberOfLaddersInBarrel)](
            fill=0
        )
        self.ladderMinZ = InlineArray[Float, Int(Self.numberOfLaddersInBarrel)](
            fill=0
        )
        self.ladderMaxZ = InlineArray[Float, Int(Self.numberOfLaddersInBarrel)](
            fill=0
        )
        self.endCapZ = InlineArray[Float, 2](fill=0)

    @always_inline
    @staticmethod
    fn dtype() -> String:
        return "AverageGeometry"
