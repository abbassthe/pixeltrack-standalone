# Mojo port of CUDADataFormats/TrackingRecHit2DHeterogeneous.{h,cc}. CUDA-only
# (C++'s GPUTraits/CPUTraits/HostTraits templating dropped, matching
# TrackSoAHeterogeneousT.mojo/SiPixelDigisCUDA.mojo).
#
# nHits == 0 leaves m_phiBinner/m_phiBinnerStorage/m_hitsLayerStart/m_iphi
# null; C++ leaves them uninitialized (no member initializer, early-return
# path never sets them).
from MojoCudaDev.CUDACore.device_unique_ptr import (
    unique_ptr as DeviceUniquePtr,
    make_device_unique,
)
from MojoCudaDev.CUDACore.host_unique_ptr import (
    unique_ptr as HostUniquePtr,
    make_host_unique,
)
from MojoCudaDev.CUDACore.allocate_device import _AllocateDeviceState
from MojoCudaDev.CUDACore.allocate_host import _AllocateHostState
from MojoCudaDev.CUDACore.copyAsync import copyAsync, copyAsyncOwned
from MojoCudaDev.CUDACore.CUDACompat import CUDAStreamType, cudaStreamDefault
from MojoCudaDev.CUDADataFormats.TrackingRecHit2DSOAView import (
    TrackingRecHit2DSOAView,
    ParamsOnGPU,
)
from MojoCudaDev.CUDADataFormats.gpuClusteringConstants import gpuClustering
from MojoCudaDev.Geometry.Phase1PixelTopology import Phase1PixelTopology
from MojoCudaDev.MojoBridge.DTypes import Float, Typeable


struct TrackingRecHit2DHeterogeneous(Movable, Typeable):
    comptime PhiBinner = TrackingRecHit2DSOAView.PhiBinner
    comptime AverageGeometry = TrackingRecHit2DSOAView.AverageGeometry
    comptime n16: UInt32 = 4
    comptime n32: UInt32 = 10

    var m_store16: DeviceUniquePtr[UInt16]
    var m_store32: DeviceUniquePtr[Float]

    var m_PhiBinnerStore: DeviceUniquePtr[Self.PhiBinner]
    var m_AverageGeometryStore: DeviceUniquePtr[Self.AverageGeometry]

    var m_view: DeviceUniquePtr[TrackingRecHit2DSOAView]

    var m_nHits: UInt32

    var m_hitsModuleStart: UnsafePointer[UInt32, MutAnyOrigin]  # forwarded from clusters, not owned

    var m_phiBinner: UnsafePointer[Self.PhiBinner, MutAnyOrigin]
    var m_phiBinnerStorage: UnsafePointer[Self.PhiBinner.index_type, MutAnyOrigin]
    var m_hitsLayerStart: UnsafePointer[UInt32, MutAnyOrigin]
    var m_iphi: UnsafePointer[Int16, MutAnyOrigin]

    fn __init__(out self):
        self.m_store16 = DeviceUniquePtr[UInt16]()
        self.m_store32 = DeviceUniquePtr[Float]()
        self.m_PhiBinnerStore = DeviceUniquePtr[Self.PhiBinner]()
        self.m_AverageGeometryStore = DeviceUniquePtr[Self.AverageGeometry]()
        self.m_view = DeviceUniquePtr[TrackingRecHit2DSOAView]()
        self.m_nHits = 0
        self.m_hitsModuleStart = UnsafePointer[UInt32, MutAnyOrigin]()
        self.m_phiBinner = UnsafePointer[Self.PhiBinner, MutAnyOrigin]()
        self.m_phiBinnerStorage = UnsafePointer[Self.PhiBinner.index_type, MutAnyOrigin]()
        self.m_hitsLayerStart = UnsafePointer[UInt32, MutAnyOrigin]()
        self.m_iphi = UnsafePointer[Int16, MutAnyOrigin]()

    fn __init__(
        out self,
        nHits: UInt32,
        cpeParams: UnsafePointer[ParamsOnGPU, MutAnyOrigin],
        hitsModuleStart: UnsafePointer[UInt32, MutAnyOrigin],
        mut dev_state: _AllocateDeviceState,
        mut host_state: _AllocateHostState,
        stream: CUDAStreamType = cudaStreamDefault,
    ) raises:
        self.m_store16 = DeviceUniquePtr[UInt16]()
        self.m_store32 = DeviceUniquePtr[Float]()
        self.m_PhiBinnerStore = DeviceUniquePtr[Self.PhiBinner]()
        self.m_AverageGeometryStore = DeviceUniquePtr[Self.AverageGeometry]()
        self.m_view = DeviceUniquePtr[TrackingRecHit2DSOAView]()
        self.m_nHits = nHits
        self.m_hitsModuleStart = hitsModuleStart
        self.m_phiBinner = UnsafePointer[Self.PhiBinner, MutAnyOrigin]()
        self.m_phiBinnerStorage = UnsafePointer[Self.PhiBinner.index_type, MutAnyOrigin]()
        self.m_hitsLayerStart = UnsafePointer[UInt32, MutAnyOrigin]()
        self.m_iphi = UnsafePointer[Int16, MutAnyOrigin]()

        var view = make_host_unique[TrackingRecHit2DSOAView](host_state, stream)
        view[].get()[0].m_nHits = nHits
        self.m_view = make_device_unique[TrackingRecHit2DSOAView](dev_state, stream)
        self.m_AverageGeometryStore = make_device_unique[Self.AverageGeometry](dev_state, stream)
        view[].get()[0].m_averageGeometry = self.m_AverageGeometryStore.get()
        view[].get()[0].m_cpeParams = cpeParams
        view[].get()[0].m_hitsModuleStart = hitsModuleStart

        if nHits == 0:
            copyAsyncOwned[TrackingRecHit2DSOAView](self.m_view, view^, stream)
            return

        self.m_store16 = make_device_unique[UInt16](UInt(nHits * Self.n16), dev_state, stream)
        self.m_store32 = make_device_unique[Float](
            UInt(nHits * Self.n32 + Phase1PixelTopology.numberOfLayers + 1), dev_state, stream
        )
        self.m_PhiBinnerStore = make_device_unique[Self.PhiBinner](dev_state, stream)

        var store16 = self.m_store16.get()
        var store32 = self.m_store32.get()
        var n = Int(nHits)

        self.m_phiBinner = self.m_PhiBinnerStore.get()
        view[].get()[0].m_phiBinner = self.m_phiBinner
        self.m_phiBinnerStorage = (store32 + 9 * n).bitcast[Self.PhiBinner.index_type]()
        view[].get()[0].m_phiBinnerStorage = self.m_phiBinnerStorage

        view[].get()[0].m_xl = store32 + 0 * n
        view[].get()[0].m_yl = store32 + 1 * n
        view[].get()[0].m_xerr = store32 + 2 * n
        view[].get()[0].m_yerr = store32 + 3 * n

        view[].get()[0].m_xg = store32 + 4 * n
        view[].get()[0].m_yg = store32 + 5 * n
        view[].get()[0].m_zg = store32 + 6 * n
        view[].get()[0].m_rg = store32 + 7 * n

        self.m_iphi = (store16 + 0 * n).bitcast[Int16]()
        view[].get()[0].m_iphi = self.m_iphi

        view[].get()[0].m_charge = (store32 + 8 * n).bitcast[Int32]()
        view[].get()[0].m_xsize = (store16 + 2 * n).bitcast[Int16]()
        view[].get()[0].m_ysize = (store16 + 3 * n).bitcast[Int16]()
        view[].get()[0].m_detInd = store16 + 1 * n

        self.m_hitsLayerStart = (store32 + Int(Self.n32) * n).bitcast[UInt32]()
        view[].get()[0].m_hitsLayerStart = self.m_hitsLayerStart

        copyAsyncOwned[TrackingRecHit2DSOAView](self.m_view, view^, stream)

    fn __moveinit__(out self, deinit take: Self):
        self.m_store16 = take.m_store16^
        self.m_store32 = take.m_store32^
        self.m_PhiBinnerStore = take.m_PhiBinnerStore^
        self.m_AverageGeometryStore = take.m_AverageGeometryStore^
        self.m_view = take.m_view^
        self.m_nHits = take.m_nHits
        self.m_hitsModuleStart = take.m_hitsModuleStart
        self.m_phiBinner = take.m_phiBinner
        self.m_phiBinnerStorage = take.m_phiBinnerStorage
        self.m_hitsLayerStart = take.m_hitsLayerStart
        self.m_iphi = take.m_iphi

    fn _get16(self, i: Int) -> UnsafePointer[UInt16, MutAnyOrigin]:
        return self.m_store16.get() + i * Int(self.m_nHits)

    fn _get32(self, i: Int) -> UnsafePointer[Float, MutAnyOrigin]:
        return self.m_store32.get() + i * Int(self.m_nHits)

    fn view(self) -> UnsafePointer[TrackingRecHit2DSOAView, MutAnyOrigin]:
        return self.m_view.get()

    fn nHits(self) -> UInt32:
        return self.m_nHits

    fn hitsModuleStart(self) -> UnsafePointer[UInt32, MutAnyOrigin]:
        return self.m_hitsModuleStart

    fn hitsLayerStart(self) -> UnsafePointer[UInt32, MutAnyOrigin]:
        return self.m_hitsLayerStart

    fn phiBinner(self) -> UnsafePointer[Self.PhiBinner, MutAnyOrigin]:
        return self.m_phiBinner

    fn phiBinnerStorage(self) -> UnsafePointer[Self.PhiBinner.index_type, MutAnyOrigin]:
        return self.m_phiBinnerStorage

    fn iphi(self) -> UnsafePointer[Int16, MutAnyOrigin]:
        return self.m_iphi

    # only the local coord and detector index
    fn localCoordToHostAsync(
        self, mut host_state: _AllocateHostState, stream: CUDAStreamType
    ) raises -> HostUniquePtr[Float]:
        var ret = make_host_unique[Float](UInt(4 * self.m_nHits), host_state, stream)
        copyAsync[Float](ret, self.m_store32, UInt(4 * self.m_nHits), stream)
        return ret^

    fn hitsModuleStartToHostAsync(
        self, mut host_state: _AllocateHostState, stream: CUDAStreamType
    ) raises -> HostUniquePtr[UInt32]:
        var n = UInt(gpuClustering.maxNumModules) + 1
        var ret = make_host_unique[UInt32](n, host_state, stream)
        copyAsync[UInt32](ret, self.m_hitsModuleStart, n, stream)
        return ret^

    # for validation
    fn globalCoordToHostAsync(
        self, mut host_state: _AllocateHostState, stream: CUDAStreamType
    ) raises -> HostUniquePtr[Float]:
        var ret = make_host_unique[Float](UInt(4 * self.m_nHits), host_state, stream)
        copyAsync[Float](ret, self._get32(4), UInt(4 * self.m_nHits), stream)
        return ret^

    fn chargeToHostAsync(
        self, mut host_state: _AllocateHostState, stream: CUDAStreamType
    ) raises -> HostUniquePtr[Int32]:
        var ret = make_host_unique[Int32](UInt(self.m_nHits), host_state, stream)
        copyAsync[Int32](ret, self._get32(8).bitcast[Int32](), UInt(self.m_nHits), stream)
        return ret^

    fn sizeToHostAsync(
        self, mut host_state: _AllocateHostState, stream: CUDAStreamType
    ) raises -> HostUniquePtr[Int16]:
        var ret = make_host_unique[Int16](UInt(2 * self.m_nHits), host_state, stream)
        copyAsync[Int16](ret, self._get16(2).bitcast[Int16](), UInt(2 * self.m_nHits), stream)
        return ret^

    @staticmethod
    fn dtype() -> String:
        return "TrackingRecHit2DHeterogeneous"
