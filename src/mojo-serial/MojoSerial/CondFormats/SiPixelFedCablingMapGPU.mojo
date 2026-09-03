from std.sys import size_of

from MojoSerial.CondFormats.PixelCPEforGPU import (
    CommonParams,
    DetParams,
    LayerGeometry,
    AverageGeometry,
)
from MojoSerial.MojoBridge.DTypes import UChar, Typeable
from MojoSerial.CondFormats.PixelGPUDetails import PixelGPUDetails


# WARNING: THIS STRUCT IS 128-ALIGNED
struct SiPixelFedCablingMapGPU(Defaultable, Movable, Typeable):
    comptime _U = InlineArray[UInt32, Int(PixelGPUDetails.MAX_SIZE)]
    comptime _UD = Self._U(uninitialized=True)
    comptime _C = InlineArray[UChar, Int(PixelGPUDetails.MAX_SIZE)]
    comptime _CD = Self._C(uninitialized=True)
    var fed: Self._U
    var link: Self._U
    var roc: Self._U
    var RawId: Self._U
    var rocInDet: Self._U
    var moduleId: Self._U
    var badRocs: Self._C
    var size: UInt32
    var __padding: InlineArray[UInt8, 124]

    @always_inline
    def __init__(out self):
        self.fed = Self._U(fill=0)
        self.link = Self._U(fill=0)
        self.roc = Self._U(fill=0)
        self.RawId = Self._U(fill=0)
        self.rocInDet = Self._U(fill=0)
        self.moduleId = Self._U(fill=0)
        self.badRocs = Self._C(fill=0)
        self.size = 0

        self.__padding = InlineArray[UInt8, 124](fill=0)

    @always_inline
    def __init__(
        out self,
        var fed: Self._U,
        var link: Self._U,
        var roc: Self._U,
        var RawId: Self._U,
        var rocInDet: Self._U,
        var moduleId: Self._U,
        var badRocs: Self._C,
    ):
        self.fed = fed^
        self.link = link^
        self.roc = roc^
        self.RawId = RawId^
        self.rocInDet = rocInDet^
        self.moduleId = moduleId^
        self.badRocs = badRocs^
        self.size = 0

        self.__padding = InlineArray[UInt8, 124](fill=0)

    @always_inline
    def __init__(out self, *, deinit move: Self):
        self.fed = move.fed^
        self.link = move.link^
        self.roc = move.roc^
        self.RawId = move.RawId^
        self.rocInDet = move.rocInDet^
        self.moduleId = move.moduleId^
        self.badRocs = move.badRocs^
        self.size = move.size

        self.__padding = InlineArray[UInt8, 124](fill=0)

    @always_inline
    @staticmethod
    def dtype() -> String:
        return "SiPixelFedCablingMapGPU"
