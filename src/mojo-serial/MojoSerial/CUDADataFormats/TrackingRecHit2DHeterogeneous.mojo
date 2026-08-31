from std.memory import OwnedPointer
from std.sys import size_of

from MojoSerial.CondFormats.PixelCPEforGPU import ParamsOnGPU
from MojoSerial.CUDACore.CUDACompat import CUDAStreamType, cudaStreamDefault
from MojoSerial.CUDADataFormats.TrackingRecHit2DSOAView import (
    Hist,
    TrackingRecHit2DSOAView,
)
from MojoSerial.Geometry.Phase1PixelTopology import (
    Phase1PixelTopology,
    AverageGeometry,
)
from MojoSerial.MojoBridge.DTypes import Float, Typeable


struct TrackingRecHit2DHeterogeneous(Defaultable, Movable, Typeable):
    comptime n16: UInt32 = 4
    comptime n32: UInt32 = 9

    comptime __d = debug_assert(
        size_of[UInt32]() == size_of[Float]()
    )  # just stating the obvious

    var m_store16: OwnedPointer[List[UInt16]]
    var m_store32: OwnedPointer[List[Float]]

    var m_HistStore: OwnedPointer[Hist]
    var m_AverageGeometryStore: OwnedPointer[AverageGeometry]

    var m_view: OwnedPointer[TrackingRecHit2DSOAView]

    var m_nHits: UInt32

    # needed for legacy
    var m_hitsModuleStart: UnsafePointer[UInt32]

    # needed as kernel params
    var m_hist: UnsafePointer[Hist]
    var m_hitsLayerStart: UnsafePointer[UInt32]
    var m_iphi: UnsafePointer[Int16]

    @always_inline
    def __init__(out self):
        self.m_store16 = OwnedPointer(List[UInt16]())
        self.m_store32 = OwnedPointer(List[Float]())
        self.m_HistStore = OwnedPointer(Hist())
        self.m_AverageGeometryStore = OwnedPointer(AverageGeometry())
        self.m_view = OwnedPointer(TrackingRecHit2DSOAView())

        self.m_nHits = 0
        self.m_hitsModuleStart = UnsafePointer[UInt32]()
        self.m_hist = self.m_HistStore.unsafe_ptr()

        self.m_hitsLayerStart = UnsafePointer[UInt32]()
        self.m_iphi = UnsafePointer[Int16]()

    def __init__(
        out self,
        nHits: UInt32,
        cpeParams: UnsafePointer[ParamsOnGPU],
        hitsModuleStart: UnsafePointer[UInt32],
        stream: CUDAStreamType = cudaStreamDefault,
    ):
        self.m_nHits = nHits
        self.m_hitsModuleStart = hitsModuleStart

        self.m_view = OwnedPointer(TrackingRecHit2DSOAView())

        self.m_view[].m_nHits = self.m_nHits
        self.m_AverageGeometryStore = OwnedPointer(AverageGeometry())
        self.m_view[].m_averageGeometry = (
            self.m_AverageGeometryStore.unsafe_ptr()
        )
        self.m_view[].m_cpeParams = cpeParams
        self.m_view[].m_hitsModuleStart = self.m_hitsModuleStart

        # if empty do not bother
        if nHits == 0:
            # must initialize remaining fields to defaults before returning
            self.m_store16 = OwnedPointer(List[UInt16]())
            self.m_store32 = OwnedPointer(List[Float]())
            self.m_HistStore = OwnedPointer(Hist())
            self.m_hitsLayerStart = UnsafePointer[UInt32]()
            self.m_iphi = UnsafePointer[Int16]()
            self.m_hist = self.m_HistStore.unsafe_ptr()
            return

        self.m_store16 = OwnedPointer(
            List[UInt16](length=Int(nHits * Self.n16), fill=0)
        )
        self.m_store32 = OwnedPointer(
            List[Float](length=Int(nHits * Self.n32 + 11), fill=0)
        )
        self.m_HistStore = OwnedPointer(Hist())

        self.m_hitsLayerStart = UnsafePointer[UInt32]()

        # cannot wrap self in an @parameter function without having all fields initialized
        self.m_hist = self.m_HistStore.unsafe_ptr()
        self.m_iphi = UnsafePointer[Int16]()

        # fyi: @parameter captures by reference
        @parameter
        def get16(i: Int) -> UnsafePointer[UInt16]:
            return self.m_store16[].unsafe_ptr() + i * nHits

        @parameter
        def get32(i: Int) -> UnsafePointer[Float]:
            return self.m_store32[].unsafe_ptr() + i * nHits

        # copy all the pointers
        self.m_view[].m_hist = self.m_hist
        self.m_view[].m_xl = get32(0)
        self.m_view[].m_yl = get32(1)
        self.m_view[].m_xerr = get32(2)
        self.m_view[].m_yerr = get32(3)

        self.m_view[].m_xg = get32(4)
        self.m_view[].m_yg = get32(5)
        self.m_view[].m_zg = get32(6)
        self.m_view[].m_rg = get32(7)

        self.m_iphi = get16(0).bitcast[Int16]()
        self.m_view[].m_iphi = self.m_iphi

        self.m_view[].m_charge = get32(8).bitcast[Int32]()
        self.m_view[].m_xsize = get16(2).bitcast[Int16]()
        self.m_view[].m_ysize = get16(3).bitcast[Int16]()
        self.m_view[].m_detInd = get16(1)

        self.m_hitsLayerStart = get32(Int(Self.n32)).bitcast[UInt32]()
        self.m_view[].m_hitsLayerStart = self.m_hitsLayerStart

    @always_inline
    def __moveinit__(out self, var other: Self):
        self.m_store16 = other.m_store16^
        self.m_store32 = other.m_store32^
        self.m_HistStore = other.m_HistStore^
        self.m_AverageGeometryStore = other.m_AverageGeometryStore^
        self.m_view = other.m_view^

        self.m_nHits = other.m_nHits
        self.m_hitsModuleStart = other.m_hitsModuleStart
        self.m_hist = other.m_hist

        self.m_hitsLayerStart = other.m_hitsLayerStart
        self.m_iphi = other.m_iphi

    @always_inline
    def view[
        origin: Origin, //
    ](ref [origin]self) -> UnsafePointer[
        TrackingRecHit2DSOAView, mut = origin.mut, origin=origin
    ]:
        return self.m_view.unsafe_ptr()

    @always_inline
    def nHits(self) -> UInt32:
        return self.m_nHits

    @always_inline
    def hitsModuleStart(self) -> UnsafePointer[UInt32, mut=False]:
        return self.m_hitsModuleStart

    @always_inline
    def hitsLayerStart(mut self) -> UnsafePointer[UInt32, mut=True]:
        return self.m_hitsLayerStart

    @always_inline
    def phiBinner(mut self) -> UnsafePointer[Hist, mut=True]:
        return self.m_hist

    @always_inline
    def iphi(mut self) -> UnsafePointer[Int16, mut=True]:
        return self.m_iphi

    @always_inline
    @staticmethod
    def dtype() -> String:
        return "TrackingRecHit2DHeterogeneous"


comptime TrackingRecHit2DCPU = TrackingRecHit2DHeterogeneous
