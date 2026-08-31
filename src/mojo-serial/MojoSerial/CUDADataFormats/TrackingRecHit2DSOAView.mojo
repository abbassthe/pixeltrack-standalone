from std.sys import size_of

from MojoSerial.CondFormats.PixelCPEforGPU import ParamsOnGPU
from MojoSerial.CUDACore.HistoContainer import HistoContainer
from MojoSerial.CUDADataFormats.GPUClusteringConstants import (
    GPUClusteringConstants,
)
from MojoSerial.Geometry.Phase1PixelTopology import (
    Phase1PixelTopology,
    AverageGeometry,
)
from MojoSerial.MojoBridge.DTypes import Float, Typeable

comptime Hist = HistoContainer[
    DType.int16,
    128,
    GPUClusteringConstants.MaxNumClusters,
    8 * size_of[UInt16](),
    DType.int16,
    10,
]


@fieldwise_init
struct TrackingRecHit2DSOAView(Defaultable, Movable, Typeable):
    @staticmethod
    @always_inline
    def maxHits() -> UInt32:
        return GPUClusteringConstants.MaxNumClusters

    comptime HIndexType = UInt16  # if above is <=2^16

    # local coord
    var m_xl: UnsafePointer[Float]
    var m_yl: UnsafePointer[Float]
    var m_xerr: UnsafePointer[Float]
    var m_yerr: UnsafePointer[Float]

    # global coord
    var m_xg: UnsafePointer[Float]
    var m_yg: UnsafePointer[Float]
    var m_zg: UnsafePointer[Float]
    var m_rg: UnsafePointer[Float]
    var m_iphi: UnsafePointer[Int16]

    # cluster properties
    var m_charge: UnsafePointer[Int32]
    var m_xsize: UnsafePointer[Int16]
    var m_ysize: UnsafePointer[Int16]
    var m_detInd: UnsafePointer[UInt16]

    # supporting objects
    var m_averageGeometry: UnsafePointer[
        AverageGeometry
    ]  # owned (corrected for beam spot: not sure where to host it otherwise)
    var m_cpeParams: UnsafePointer[
        ParamsOnGPU
    ]  # forwarded from setup, NOT owned
    var m_hitsModuleStart: UnsafePointer[UInt32]  # forwarded from clusters

    var m_hitsLayerStart: UnsafePointer[UInt32]
    var m_hist: UnsafePointer[Hist]
    var m_nHits: UInt32

    def __init__(out self):
        self.m_xl = UnsafePointer[Float]()
        self.m_yl = UnsafePointer[Float]()
        self.m_xerr = UnsafePointer[Float]()
        self.m_yerr = UnsafePointer[Float]()

        self.m_xg = UnsafePointer[Float]()
        self.m_yg = UnsafePointer[Float]()
        self.m_zg = UnsafePointer[Float]()
        self.m_rg = UnsafePointer[Float]()
        self.m_iphi = UnsafePointer[Int16]()

        self.m_charge = UnsafePointer[Int32]()
        self.m_xsize = UnsafePointer[Int16]()
        self.m_ysize = UnsafePointer[Int16]()
        self.m_detInd = UnsafePointer[UInt16]()

        self.m_averageGeometry = UnsafePointer[AverageGeometry]()
        self.m_cpeParams = UnsafePointer[ParamsOnGPU]()
        self.m_hitsModuleStart = UnsafePointer[UInt32]()

        self.m_hitsLayerStart = UnsafePointer[UInt32]()
        self.m_hist = UnsafePointer[Hist]()
        self.m_nHits = 0

    @always_inline
    def nHits(self) -> UInt32:
        return self.m_nHits

    @always_inline
    def xLocal(ref self, i: Int) -> ref [self.m_xl] Float:
        return self.m_xl[i]

    @always_inline
    def yLocal(ref self, i: Int) -> ref [self.m_yl] Float:
        return self.m_yl[i]

    @always_inline
    def xerrLocal(ref self, i: Int) -> ref [self.m_xerr] Float:
        return self.m_xerr[i]

    @always_inline
    def yerrLocal(ref self, i: Int) -> ref [self.m_yerr] Float:
        return self.m_yerr[i]

    @always_inline
    def xGlobal(ref self, i: Int) -> ref [self.m_xg] Float:
        return self.m_xg[i]

    @always_inline
    def yGlobal(ref self, i: Int) -> ref [self.m_yg] Float:
        return self.m_yg[i]

    @always_inline
    def zGlobal(ref self, i: Int) -> ref [self.m_zg] Float:
        return self.m_zg[i]

    @always_inline
    def rGlobal(ref self, i: Int) -> ref [self.m_rg] Float:
        return self.m_rg[i]

    @always_inline
    def iphi(ref self, i: Int) -> ref [self.m_iphi] Int16:
        return self.m_iphi[i]

    @always_inline
    def charge(ref self, i: Int) -> ref [self.m_charge] Int32:
        return self.m_charge[i]

    @always_inline
    def clusterSizeX(ref self, i: Int) -> ref [self.m_xsize] Int16:
        return self.m_xsize[i]

    @always_inline
    def clusterSizeY(ref self, i: Int) -> ref [self.m_ysize] Int16:
        return self.m_ysize[i]

    @always_inline
    def detectorIndex(ref self, i: Int) -> ref [self.m_detInd] UInt16:
        return self.m_detInd[i]

    @always_inline
    def cpeParams(self) -> ref [self.m_cpeParams] ParamsOnGPU:
        return self.m_cpeParams[]

    @always_inline
    def hitsModuleStart(self, i: Int) -> UInt32:
        return self.m_hitsModuleStart[i]

    @always_inline
    def hitsLayerStart[
        origin: Origin, //
    ](ref [origin]self) -> UnsafePointer[
        UInt32, mut = origin.mut, origin=origin
    ]:
        return self.m_hitsLayerStart

    @always_inline
    def phiBinner(ref self) -> ref [self.m_hist] Hist:
        return self.m_hist[]

    @always_inline
    def averageGeometry(
        ref self,
    ) -> ref [self.m_averageGeometry] AverageGeometry:
        return self.m_averageGeometry[]

    @always_inline
    @staticmethod
    def dtype() -> String:
        return "TrackingRecHit2DSOAView"
