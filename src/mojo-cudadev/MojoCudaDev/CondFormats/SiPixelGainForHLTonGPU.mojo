# Mojo port of CondFormats/SiPixelGainForHLTonGPU.h. Pure rename sweep from
# `cuda` (confirmed via diff): v_pedestals/rangeAndCols/pedPrecision/
# gainPrecision all gained a trailing underscore, and the hardcoded array
# size 2000 became gpuClustering::maxNumModules -- no logic changes.
from MojoCudaDev.CUDADataFormats.gpuClusteringConstants import gpuClustering
from MojoCudaDev.MojoBridge.DTypes import Float, Typeable


@fieldwise_init
struct SiPixelGainForHLTonGPU_DecodingStructure(
    Copyable, Defaultable, Movable, Typeable, TrivialRegisterPassable
):
    var gain: UInt8
    var ped: UInt8

    fn __init__(out self):
        self.gain = 0
        self.ped = 0

    @staticmethod
    fn dtype() -> String:
        return "SiPixelGainForHLTonGPU_DecodingStructure"


@fieldwise_init
struct SiPixelGainForHLTonGPU(Copyable, Defaultable, Movable, Typeable):
    comptime DecodingStructure = SiPixelGainForHLTonGPU_DecodingStructure
    comptime Range = Tuple[UInt32, UInt32]

    var v_pedestals_: UnsafePointer[Self.DecodingStructure, MutAnyOrigin]
    var rangeAndCols_: InlineArray[Tuple[Self.Range, Int32], Int(gpuClustering.maxNumModules)]
    var minPed_: Float
    var maxPed_: Float
    var minGain_: Float
    var maxGain_: Float

    var pedPrecision_: Float
    var gainPrecision_: Float

    var numberOfRowsAveragedOver_: UInt32  # this is 80!!!!
    var nBinsToUseForEncoding_: UInt32
    var deadFlag_: UInt32
    var noisyFlag_: UInt32

    fn __init__(out self):
        self.v_pedestals_ = UnsafePointer[Self.DecodingStructure, MutAnyOrigin]()
        self.rangeAndCols_ = InlineArray[Tuple[Self.Range, Int32], Int(gpuClustering.maxNumModules)](
            fill=Tuple[Self.Range, Int32](Self.Range(0, 0), 0)
        )
        self.minPed_ = 0.0
        self.maxPed_ = 0.0
        self.minGain_ = 0.0
        self.maxGain_ = 0.0

        self.pedPrecision_ = 0.0
        self.gainPrecision_ = 0.0

        self.numberOfRowsAveragedOver_ = 0
        self.nBinsToUseForEncoding_ = 0
        self.deadFlag_ = 0
        self.noisyFlag_ = 0

    fn getPedAndGain(
        self,
        moduleInd: UInt32,
        col: Int32,
        row: Int32,
        mut isDeadColumn: Bool,
        mut isNoisyColumn: Bool,
    ) -> Tuple[Float, Float]:
        var range = self.rangeAndCols_[moduleInd][0]
        var nCols = self.rangeAndCols_[moduleInd][1]

        # determine what averaged data block we are in (there should be 1 or 2 of these depending on if plaquette is 1 by X or 2 by X
        var lengthOfColumnData: UInt32 = (
            (range[1].cast[DType.int32]() - range[0].cast[DType.int32]()) // nCols
        ).cast[DType.uint32]()
        # we always only have two values per column averaged block
        var lengthOfAveragedDataInEachColumn: UInt32 = 2
        var numberOfDataBlocksToSkip = row.cast[DType.uint32]() // self.numberOfRowsAveragedOver_
        var offset = (
            range[0]
            + col.cast[DType.uint32]() * lengthOfColumnData
            + lengthOfAveragedDataInEachColumn * numberOfDataBlocksToSkip
        )
        debug_assert(offset < range[1])
        debug_assert(offset < 3088384)
        debug_assert(offset % 2 == 0)

        var lp = self.v_pedestals_
        var s = lp[offset // 2]

        isDeadColumn = (s.ped.cast[DType.uint32]() & 0xFF) == self.deadFlag_
        isNoisyColumn = (s.ped.cast[DType.uint32]() & 0xFF) == self.noisyFlag_
        return (
            self.decodePed(s.ped.cast[DType.uint32]() & 0xFF),
            self.decodeGain(s.gain.cast[DType.uint32]() & 0xFF),
        )

    fn decodeGain(self, gain: UInt32) -> Float:
        return gain.cast[DType.float32]() * self.gainPrecision_ + self.minGain_

    fn decodePed(self, ped: UInt32) -> Float:
        return ped.cast[DType.float32]() * self.pedPrecision_ + self.minPed_

    @staticmethod
    fn dtype() -> String:
        return "SiPixelGainForHLTonGPU"
