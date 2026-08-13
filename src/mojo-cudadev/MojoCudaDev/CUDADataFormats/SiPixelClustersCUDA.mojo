# Mojo port of CUDADataFormats/SiPixelClustersCUDA.{h,cc}. Four device arrays
# plus a DeviceConstView of them, staged on the host and copied to the device.
# The allocating constructor takes explicit allocator-state params that C++'s
# doesn't need (C++ reaches global allocators implicitly) -- same deviation as
# HeterogeneousSoA.toHostAsync and BeamSpotCUDA.
from MojoCudaDev.CUDACore.device_unique_ptr import (
    unique_ptr as DeviceUniquePtr,
    make_device_unique,
)
from MojoCudaDev.CUDACore.host_unique_ptr import make_host_unique
from MojoCudaDev.CUDACore.allocate_device import _AllocateDeviceState
from MojoCudaDev.CUDACore.allocate_host import _AllocateHostState
from MojoCudaDev.CUDACore.copyAsync import copyAsync
from MojoCudaDev.CUDACore.CUDACompat import CUDAStreamType, cudaStreamDefault


# C++: __device__ __forceinline__ accessors, read through __ldg (a load hint
# with no correctness effect) -- SiPixelClustersCUDA.h:37-40
struct DeviceConstView(Movable):
    var moduleStart_: UnsafePointer[UInt32, MutAnyOrigin]
    var clusInModule_: UnsafePointer[UInt32, MutAnyOrigin]
    var moduleId_: UnsafePointer[UInt32, MutAnyOrigin]
    var clusModuleStart_: UnsafePointer[UInt32, MutAnyOrigin]

    fn __init__(out self):
        self.moduleStart_ = UnsafePointer[UInt32, MutAnyOrigin]()
        self.clusInModule_ = UnsafePointer[UInt32, MutAnyOrigin]()
        self.moduleId_ = UnsafePointer[UInt32, MutAnyOrigin]()
        self.clusModuleStart_ = UnsafePointer[UInt32, MutAnyOrigin]()

    @always_inline
    fn moduleStart(self, i: Int) -> UInt32:
        return self.moduleStart_[i]

    @always_inline
    fn clusInModule(self, i: Int) -> UInt32:
        return self.clusInModule_[i]

    @always_inline
    fn moduleId(self, i: Int) -> UInt32:
        return self.moduleId_[i]

    @always_inline
    fn clusModuleStart(self, i: Int) -> UInt32:
        return self.clusModuleStart_[i]


struct SiPixelClustersCUDA(Movable):
    var moduleStart_d: DeviceUniquePtr[UInt32]  # index of the first pixel of each module
    var clusInModule_d: DeviceUniquePtr[UInt32]  # number of clusters found in each module
    var moduleId_d: DeviceUniquePtr[UInt32]  # module id of each module

    # originally from rechits
    var clusModuleStart_d: DeviceUniquePtr[UInt32]  # index of the first cluster of each module

    var view_d: DeviceUniquePtr[DeviceConstView]  # "me" pointer

    var nClusters_h: UInt32

    fn __init__(out self):
        self.moduleStart_d = DeviceUniquePtr[UInt32]()
        self.clusInModule_d = DeviceUniquePtr[UInt32]()
        self.moduleId_d = DeviceUniquePtr[UInt32]()
        self.clusModuleStart_d = DeviceUniquePtr[UInt32]()
        self.view_d = DeviceUniquePtr[DeviceConstView]()
        self.nClusters_h = 0

    fn __init__(
        out self,
        maxModules: UInt,
        mut dev_state: _AllocateDeviceState,
        mut host_state: _AllocateHostState,
        stream: CUDAStreamType = cudaStreamDefault,
    ) raises:
        self.moduleStart_d = make_device_unique[UInt32](maxModules + 1, dev_state, stream)
        self.clusInModule_d = make_device_unique[UInt32](maxModules, dev_state, stream)
        self.moduleId_d = make_device_unique[UInt32](maxModules, dev_state, stream)
        self.clusModuleStart_d = make_device_unique[UInt32](maxModules + 1, dev_state, stream)
        self.nClusters_h = 0

        var view = make_host_unique[DeviceConstView](host_state, stream)
        view[].get()[0].moduleStart_ = self.moduleStart_d.get()
        view[].get()[0].clusInModule_ = self.clusInModule_d.get()
        view[].get()[0].moduleId_ = self.moduleId_d.get()
        view[].get()[0].clusModuleStart_ = self.clusModuleStart_d.get()

        self.view_d = make_device_unique[DeviceConstView](dev_state, stream)
        copyAsync[DeviceConstView](self.view_d, view, stream)
        # C++'s caching host allocator defers the free until the stream catches
        # up; this port's frees on the spot, so the staging buffer would go away
        # under the copy without this.
        stream.synchronize()

    fn __moveinit__(out self, deinit take: Self):
        self.moduleStart_d = take.moduleStart_d^
        self.clusInModule_d = take.clusInModule_d^
        self.moduleId_d = take.moduleId_d^
        self.clusModuleStart_d = take.clusModuleStart_d^
        self.view_d = take.view_d^
        self.nClusters_h = take.nClusters_h

    fn setNClusters(mut self, nClusters: UInt32):
        self.nClusters_h = nClusters

    fn nClusters(self) -> UInt32:
        return self.nClusters_h

    fn moduleStart(mut self) -> UnsafePointer[UInt32, MutAnyOrigin]:
        return self.moduleStart_d.get()

    fn clusInModule(mut self) -> UnsafePointer[UInt32, MutAnyOrigin]:
        return self.clusInModule_d.get()

    fn moduleId(mut self) -> UnsafePointer[UInt32, MutAnyOrigin]:
        return self.moduleId_d.get()

    fn clusModuleStart(mut self) -> UnsafePointer[UInt32, MutAnyOrigin]:
        return self.clusModuleStart_d.get()

    fn view(self) -> UnsafePointer[DeviceConstView, MutAnyOrigin]:
        return self.view_d.get()
