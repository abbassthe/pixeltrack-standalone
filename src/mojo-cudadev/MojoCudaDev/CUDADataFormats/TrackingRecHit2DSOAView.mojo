# Mojo port of CUDADataFormats/TrackingRecHit2DSOAView.h. A plain view struct,
# no constructor in C++ either -- fields are written directly by
# TrackingRecHit2DHeterogeneous's constructor. __ldg (C++'s separate `const`
# read-only overloads) is a load hint with no correctness effect, dropped in
# favor of this port's established single ref-returning accessor per field.
from MojoCudaDev.CondFormats.PixelCPEforGPU import ParamsOnGPU
from MojoCudaDev.CUDACore.HistoContainer import HistoContainer
from MojoCudaDev.Geometry.Phase1PixelTopology import AverageGeometry as _AverageGeometry
from MojoCudaDev.MojoBridge.DTypes import Float, Typeable


struct TrackingRecHit2DSOAView(Movable, Typeable):
    comptime hindex_type = UInt32
    comptime PhiBinner = HistoContainer[DType.int16, 128, -1, 16, Self.hindex_type, 10]
    comptime AverageGeometry = _AverageGeometry

    # local coord
    var m_xl: UnsafePointer[Float, MutAnyOrigin]
    var m_yl: UnsafePointer[Float, MutAnyOrigin]
    var m_xerr: UnsafePointer[Float, MutAnyOrigin]
    var m_yerr: UnsafePointer[Float, MutAnyOrigin]

    # global coord
    var m_xg: UnsafePointer[Float, MutAnyOrigin]
    var m_yg: UnsafePointer[Float, MutAnyOrigin]
    var m_zg: UnsafePointer[Float, MutAnyOrigin]
    var m_rg: UnsafePointer[Float, MutAnyOrigin]
    var m_iphi: UnsafePointer[Int16, MutAnyOrigin]

    # cluster properties
    var m_charge: UnsafePointer[Int32, MutAnyOrigin]
    var m_xsize: UnsafePointer[Int16, MutAnyOrigin]
    var m_ysize: UnsafePointer[Int16, MutAnyOrigin]
    var m_detInd: UnsafePointer[UInt16, MutAnyOrigin]

    # supporting objects
    var m_averageGeometry: UnsafePointer[Self.AverageGeometry, MutAnyOrigin]  # owned by TrackingRecHit2DHeterogeneous
    var m_cpeParams: UnsafePointer[ParamsOnGPU, MutAnyOrigin]  # forwarded from setup, NOT owned
    var m_hitsModuleStart: UnsafePointer[UInt32, MutAnyOrigin]  # forwarded from clusters

    var m_hitsLayerStart: UnsafePointer[UInt32, MutAnyOrigin]

    var m_phiBinner: UnsafePointer[Self.PhiBinner, MutAnyOrigin]
    var m_phiBinnerStorage: UnsafePointer[Self.PhiBinner.index_type, MutAnyOrigin]

    var m_nHits: UInt32

    fn __init__(out self):
        self.m_xl = UnsafePointer[Float, MutAnyOrigin]()
        self.m_yl = UnsafePointer[Float, MutAnyOrigin]()
        self.m_xerr = UnsafePointer[Float, MutAnyOrigin]()
        self.m_yerr = UnsafePointer[Float, MutAnyOrigin]()
        self.m_xg = UnsafePointer[Float, MutAnyOrigin]()
        self.m_yg = UnsafePointer[Float, MutAnyOrigin]()
        self.m_zg = UnsafePointer[Float, MutAnyOrigin]()
        self.m_rg = UnsafePointer[Float, MutAnyOrigin]()
        self.m_iphi = UnsafePointer[Int16, MutAnyOrigin]()
        self.m_charge = UnsafePointer[Int32, MutAnyOrigin]()
        self.m_xsize = UnsafePointer[Int16, MutAnyOrigin]()
        self.m_ysize = UnsafePointer[Int16, MutAnyOrigin]()
        self.m_detInd = UnsafePointer[UInt16, MutAnyOrigin]()
        self.m_averageGeometry = UnsafePointer[Self.AverageGeometry, MutAnyOrigin]()
        self.m_cpeParams = UnsafePointer[ParamsOnGPU, MutAnyOrigin]()
        self.m_hitsModuleStart = UnsafePointer[UInt32, MutAnyOrigin]()
        self.m_hitsLayerStart = UnsafePointer[UInt32, MutAnyOrigin]()
        self.m_phiBinner = UnsafePointer[Self.PhiBinner, MutAnyOrigin]()
        self.m_phiBinnerStorage = UnsafePointer[Self.PhiBinner.index_type, MutAnyOrigin]()
        self.m_nHits = 0

    fn nHits(self) -> UInt32:
        return self.m_nHits

    fn xLocal(ref self, i: Int) -> ref [self.m_xl] Float:
        return self.m_xl[i]

    fn yLocal(ref self, i: Int) -> ref [self.m_yl] Float:
        return self.m_yl[i]

    fn xerrLocal(ref self, i: Int) -> ref [self.m_xerr] Float:
        return self.m_xerr[i]

    fn yerrLocal(ref self, i: Int) -> ref [self.m_yerr] Float:
        return self.m_yerr[i]

    fn xGlobal(ref self, i: Int) -> ref [self.m_xg] Float:
        return self.m_xg[i]

    fn yGlobal(ref self, i: Int) -> ref [self.m_yg] Float:
        return self.m_yg[i]

    fn zGlobal(ref self, i: Int) -> ref [self.m_zg] Float:
        return self.m_zg[i]

    fn rGlobal(ref self, i: Int) -> ref [self.m_rg] Float:
        return self.m_rg[i]

    fn iphi(ref self, i: Int) -> ref [self.m_iphi] Int16:
        return self.m_iphi[i]

    fn charge(ref self, i: Int) -> ref [self.m_charge] Int32:
        return self.m_charge[i]

    fn clusterSizeX(ref self, i: Int) -> ref [self.m_xsize] Int16:
        return self.m_xsize[i]

    fn clusterSizeY(ref self, i: Int) -> ref [self.m_ysize] Int16:
        return self.m_ysize[i]

    fn detectorIndex(ref self, i: Int) -> ref [self.m_detInd] UInt16:
        return self.m_detInd[i]

    fn cpeParams(ref self) -> ref [self.m_cpeParams] ParamsOnGPU:
        return self.m_cpeParams[]

    fn hitsModuleStart(self, i: Int) -> UInt32:
        return self.m_hitsModuleStart[i]

    fn hitsLayerStart(self) -> UnsafePointer[UInt32, MutAnyOrigin]:
        return self.m_hitsLayerStart

    fn phiBinner(ref self) -> ref [self.m_phiBinner] Self.PhiBinner:
        return self.m_phiBinner[]

    fn averageGeometry(ref self) -> ref [self.m_averageGeometry] Self.AverageGeometry:
        return self.m_averageGeometry[]

    @staticmethod
    fn dtype() -> String:
        return "TrackingRecHit2DSOAView"
