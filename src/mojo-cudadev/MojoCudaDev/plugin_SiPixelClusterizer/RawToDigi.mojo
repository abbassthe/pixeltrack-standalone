# Mojo port of the RawToDigi helpers and kernel in
# SiPixelRawToClusterGPUKernel.cu. The public kernel uses Int32 flags because
# Mojo kernel launches do not accept Bool arguments in this toolchain.
from MojoCudaDev.CUDACore.SimpleVector import SimpleVector
from MojoCudaDev.CUDADataFormats.gpuClusteringConstants import gpuClustering
from MojoCudaDev.CondFormats.SiPixelROCsStatusAndMapping import (
    MAX_LINK,
    MAX_ROC,
    SiPixelROCsStatusAndMapping,
)
from MojoCudaDev.DataFormats.SiPixelDigiConstants import (
    SiPixelConstants,
    getADC,
    getCol,
    getDCol,
    getLink,
    getPxId,
    getROC,
    getRow,
)
from MojoCudaDev.DataFormats.SiPixelErrorCompact import SiPixelErrorCompact


@fieldwise_init
struct DetIdGPU(Copyable, Defaultable, Movable, TrivialRegisterPassable):
    var rawId: UInt32
    var rocInDet: UInt32
    var moduleId: UInt32

    fn __init__(out self):
        self.rawId = 0
        self.rocInDet = 0
        self.moduleId = 0


@fieldwise_init
struct Pixel(Copyable, Defaultable, Movable, TrivialRegisterPassable):
    var row: UInt32
    var col: UInt32

    fn __init__(out self):
        self.row = 0
        self.col = 0


struct RawToDigi:
    comptime maxROCIndex: UInt32 = 8
    comptime numRowsInRoc: UInt32 = 80
    comptime numColsInRoc: UInt32 = 52

    comptime layerStartBit: UInt32 = 20
    comptime moduleStartBit: UInt32 = 2
    comptime panelStartBit: UInt32 = 10
    comptime layerMask: UInt32 = 0xF
    comptime moduleMask: UInt32 = 0x3FF
    comptime panelMask: UInt32 = 0x3

    @always_inline
    @staticmethod
    fn pack(row: UInt32, col: UInt32, adc: UInt32) -> UInt32:
        # C++ Packing(11, 11, 0, 10).
        var clipped_adc = UInt32(1023) if adc > UInt32(1023) else adc
        return row | (col << 11) | (clipped_adc << 22)

    @always_inline
    @staticmethod
    fn isBarrel(rawId: UInt32) -> Bool:
        # PixelSubdetector::PixelBarrel == 1, DetId's subdetector field is
        # bits [27:25].
        return ((rawId >> 25) & 0x7) == 1

    @always_inline
    @staticmethod
    fn getRawId(
        cablingMap: UnsafePointer[SiPixelROCsStatusAndMapping, MutAnyOrigin],
        fed: UInt8,
        link: UInt32,
        roc: UInt32,
    ) -> DetIdGPU:
        var index = (
            Int(fed) * Int(MAX_LINK) * Int(MAX_ROC)
            + Int(link - 1) * Int(MAX_ROC)
            + Int(roc)
        )
        var result = DetIdGPU()
        result.rawId = cablingMap[].rawId[index]
        result.rocInDet = cablingMap[].rocInDet[index]
        result.moduleId = cablingMap[].moduleId[index]
        return result

    @always_inline
    @staticmethod
    fn frameConversion(
        bpix: Bool,
        side: Int32,
        layer: UInt32,
        rocIdInDetUnit: UInt32,
        local: Pixel,
    ) -> Pixel:
        var slopeRow: Int32 = 0
        var slopeCol: Int32 = 0
        var rowOffset: Int32 = 0
        var colOffset: Int32 = 0

        if bpix:
            if side == -1 and layer != 1:
                if rocIdInDetUnit < 8:
                    slopeRow = 1
                    slopeCol = -1
                    colOffset = (8 - Int32(rocIdInDetUnit)) * Int32(Self.numColsInRoc) - 1
                else:
                    slopeRow = -1
                    slopeCol = 1
                    rowOffset = 2 * Int32(Self.numRowsInRoc) - 1
                    colOffset = (Int32(rocIdInDetUnit) - 8) * Int32(Self.numColsInRoc)
            else:
                if rocIdInDetUnit < 8:
                    slopeRow = -1
                    slopeCol = 1
                    rowOffset = 2 * Int32(Self.numRowsInRoc) - 1
                    colOffset = Int32(rocIdInDetUnit) * Int32(Self.numColsInRoc)
                else:
                    slopeRow = 1
                    slopeCol = -1
                    colOffset = (16 - Int32(rocIdInDetUnit)) * Int32(Self.numColsInRoc) - 1
        else:
            if rocIdInDetUnit < 8:
                slopeRow = 1
                slopeCol = -1
                colOffset = (8 - Int32(rocIdInDetUnit)) * Int32(Self.numColsInRoc) - 1
            else:
                slopeRow = -1
                slopeCol = 1
                rowOffset = 2 * Int32(Self.numRowsInRoc) - 1
                colOffset = (Int32(rocIdInDetUnit) - 8) * Int32(Self.numColsInRoc)

        var globalPix = Pixel()
        globalPix.row = UInt32(rowOffset + slopeRow * Int32(local.row))
        globalPix.col = UInt32(colOffset + slopeCol * Int32(local.col))
        return globalPix

    @always_inline
    @staticmethod
    fn conversionError(status: UInt8) -> UInt8:
        if status == 1:
            return 35
        if status == 2:
            return 36
        if status == 3:
            return 37
        if status == 4:
            return 38
        return 0

    @always_inline
    @staticmethod
    fn rocRowColIsValid(rocRow: UInt32, rocCol: UInt32) -> Bool:
        return rocRow < Self.numRowsInRoc and rocCol < Self.numColsInRoc

    @always_inline
    @staticmethod
    fn dcolIsValid(dcol: UInt32, pxid: UInt32) -> Bool:
        return dcol < 26 and pxid >= 2 and pxid < 162

    @always_inline
    @staticmethod
    fn checkROC(
        errorWord: UInt32,
        fedId: UInt8,
        link: UInt32,
        cablingMap: UnsafePointer[SiPixelROCsStatusAndMapping, MutAnyOrigin],
    ) -> UInt8:
        var errorType = UInt8(
            (errorWord >> SiPixelConstants.ROC_shift)
            & SiPixelConstants.ERROR_mask
        )
        if errorType < 25:
            return 0

        var found = False
        if errorType == 25:
            found = True
            var index = (
                Int(fedId) * Int(MAX_LINK) * Int(MAX_ROC)
                + Int(link - 1) * Int(MAX_ROC)
                + 1
            )
            if index > 1 and UInt32(index) <= cablingMap[].size:
                if not (
                    link == cablingMap[].link[index]
                    and cablingMap[].roc[index] == 1
                ):
                    found = False
        elif errorType == 26 or errorType == 27 or errorType == 28:
            found = True
        elif errorType == 29:
            found = True
        elif errorType == 30:
            var stateMatch = Int32((errorWord >> 8) & 0xF)
            if stateMatch == 1:
                errorType = 40
            found = True
        elif errorType == 31:
            found = True

        return errorType if found else UInt8(0)

    @always_inline
    @staticmethod
    fn getErrRawID(
        fedId: UInt8,
        errWord: UInt32,
        errorType: UInt8,
        cablingMap: UnsafePointer[SiPixelROCsStatusAndMapping, MutAnyOrigin],
    ) -> UInt32:
        var rawId = UInt32(0xFFFFFFFF)
        if (
            errorType == 25 or errorType == 30 or errorType == 31
            or errorType == 36 or errorType == 40
        ):
            var mapped = Self.getRawId(cablingMap, fedId, getLink(errWord), 1).rawId
            if mapped != UInt32(gpuClustering.invalidModuleId):
                rawId = mapped
        elif errorType == 29:
            var ch1 = Int32((errWord >> 0) & 1)
            var ch2 = Int32((errWord >> 1) & 1)
            var ch3 = Int32((errWord >> 2) & 1)
            var ch4 = Int32((errWord >> 3) & 1)
            var ch5 = Int32((errWord >> 4) & 1)
            var block = Int32((errWord >> 8) & 0x7)
            var localCh = ch1 + 2 * ch2 + 3 * ch3 + 4 * ch4 + 5 * ch5
            var channel = (
                (block // 2) * 9 + localCh
                if block % 2 == 0
                else ((block - 1) // 2) * 9 + 4 + localCh
            )
            if channel >= 1 and channel <= 36:
                var mapped = Self.getRawId(cablingMap, fedId, UInt32(channel), 1).rawId
                if mapped != UInt32(gpuClustering.invalidModuleId):
                    rawId = mapped
        elif errorType == 37 or errorType == 38:
            var mapped = Self.getRawId(
                cablingMap, fedId, getLink(errWord), getROC(errWord)
            ).rawId
            if mapped != UInt32(gpuClustering.invalidModuleId):
                rawId = mapped
        return rawId

    @always_inline
    @staticmethod
    fn pushError(
        errors: UnsafePointer[SimpleVector[SiPixelErrorCompact, "SiPixelErrorCompact"], MutAnyOrigin],
        rawId: UInt32,
        word: UInt32,
        errorType: UInt8,
        fedId: UInt8,
    ):
        var error = SiPixelErrorCompact()
        error.rawId = rawId
        error.word = word
        error.errorType = errorType
        error.fedId = fedId
        _ = errors[].push_back(error)

    # C++: RawToDigi_kernel(). `useQualityInfo`, `includeErrors`, and `debug`
    # are Int32 flags solely for Mojo DevicePassable compatibility.
    @staticmethod
    fn convert(
        cablingMap: UnsafePointer[SiPixelROCsStatusAndMapping, MutAnyOrigin],
        modToUnp: UnsafePointer[UInt8, MutAnyOrigin],
        wordCounter: UInt32,
        word: UnsafePointer[UInt32, MutAnyOrigin],
        fedIds: UnsafePointer[UInt8, MutAnyOrigin],
        xx: UnsafePointer[UInt16, MutAnyOrigin],
        yy: UnsafePointer[UInt16, MutAnyOrigin],
        adc: UnsafePointer[UInt16, MutAnyOrigin],
        pdigi: UnsafePointer[UInt32, MutAnyOrigin],
        rawIdArr: UnsafePointer[UInt32, MutAnyOrigin],
        moduleId: UnsafePointer[UInt16, MutAnyOrigin],
        errors: UnsafePointer[SimpleVector[SiPixelErrorCompact, "SiPixelErrorCompact"], MutAnyOrigin],
        useQualityInfo: Int32,
        includeErrors: Int32,
        debug: Int32,
    ):
        from std.gpu import block_dim, block_idx, grid_dim, thread_idx

        _ = debug
        var first = Int(block_dim.x) * Int(block_idx.x) + Int(thread_idx.x)
        var stride = Int(block_dim.x) * Int(grid_dim.x)
        var i = first
        while i < Int(wordCounter):
            xx[i] = 0
            yy[i] = 0
            adc[i] = 0
            pdigi[i] = 0
            rawIdArr[i] = 0
            moduleId[i] = gpuClustering.invalidModuleId

            var ww = word[i]
            if ww != 0:
                var fedId = fedIds[i // 2]
                var link = getLink(ww)
                var roc = getROC(ww)
                var detId = Self.getRawId(cablingMap, fedId, link, roc)
                var errorType = Self.checkROC(ww, fedId, link, cablingMap)
                var valid = True

                if roc >= Self.maxROCIndex and errorType != 0:
                    if includeErrors != 0:
                        Self.pushError(
                            errors,
                            Self.getErrRawID(cablingMap=cablingMap, fedId=fedId, errWord=ww, errorType=errorType),
                            ww,
                            errorType,
                            fedId,
                        )
                    valid = False

                var rawId = detId.rawId
                if valid:
                    var index = (
                        Int(fedId) * Int(MAX_LINK) * Int(MAX_ROC)
                        + Int(link - 1) * Int(MAX_ROC)
                        + Int(roc)
                    )
                    if useQualityInfo != 0 and cablingMap[].badRocs[index] != 0:
                        valid = False
                    if valid and modToUnp[index] != 0:
                        valid = False

                var layer = UInt32(0)
                var side = Int32(0)
                if valid:
                    if Self.isBarrel(rawId):
                        layer = (rawId >> Self.layerStartBit) & Self.layerMask
                        var module = Int32((rawId >> Self.moduleStartBit) & Self.moduleMask)
                        side = -1 if module < 5 else 1
                    else:
                        var panel = Int32((rawId >> Self.panelStartBit) & Self.panelMask)
                        side = -1 if panel == 1 else 1

                    var local = Pixel()
                    if layer == 1:
                        local.row = getRow(ww)
                        local.col = getCol(ww)
                        if includeErrors != 0 and not Self.rocRowColIsValid(local.row, local.col):
                            Self.pushError(
                                errors, rawId, ww, Self.conversionError(3), fedId
                            )
                            valid = False
                    else:
                        var dcol = getDCol(ww)
                        var pxid = getPxId(ww)
                        local.row = Self.numRowsInRoc - pxid // 2
                        local.col = dcol * 2 + pxid % 2
                        if includeErrors != 0 and not Self.dcolIsValid(dcol, pxid):
                            Self.pushError(
                                errors, rawId, ww, Self.conversionError(3), fedId
                            )
                            valid = False

                    if valid:
                        var globalPix = Self.frameConversion(
                            Self.isBarrel(rawId), side, layer, detId.rocInDet, local
                        )
                        xx[i] = UInt16(globalPix.row)
                        yy[i] = UInt16(globalPix.col)
                        adc[i] = UInt16(getADC(ww))
                        pdigi[i] = Self.pack(globalPix.row, globalPix.col, UInt32(adc[i]))
                        moduleId[i] = UInt16(detId.moduleId)
                        rawIdArr[i] = rawId

            i += stride
