# Mojo port of CUDADataFormats/SiPixelDigisCUDA.h. The real constructor takes
# explicit device/host allocator-state params that C++ doesn't need, matching
# the pattern established by BeamSpotCUDA.mojo/HeterogeneousSoA.toHostAsync;
# it needs both because it stages DeviceConstView on the host before copying
# it to device, same as C++'s own constructor body does.
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


# C++: __device__ __forceinline__ accessors -- SiPixelDigisCUDA.h:53-58
struct DeviceConstView(Movable):
    var xx_: UnsafePointer[UInt16, MutAnyOrigin]
    var yy_: UnsafePointer[UInt16, MutAnyOrigin]
    var adc_: UnsafePointer[UInt16, MutAnyOrigin]
    var moduleInd_: UnsafePointer[UInt16, MutAnyOrigin]
    var clus_: UnsafePointer[Int32, MutAnyOrigin]

    fn __init__(out self):
        self.xx_ = UnsafePointer[UInt16, MutAnyOrigin]()
        self.yy_ = UnsafePointer[UInt16, MutAnyOrigin]()
        self.adc_ = UnsafePointer[UInt16, MutAnyOrigin]()
        self.moduleInd_ = UnsafePointer[UInt16, MutAnyOrigin]()
        self.clus_ = UnsafePointer[Int32, MutAnyOrigin]()

    @always_inline
    fn xx(self, i: Int) -> UInt16:
        return self.xx_[i]

    @always_inline
    fn yy(self, i: Int) -> UInt16:
        return self.yy_[i]

    @always_inline
    fn adc(self, i: Int) -> UInt16:
        return self.adc_[i]

    @always_inline
    fn moduleInd(self, i: Int) -> UInt16:
        return self.moduleInd_[i]

    @always_inline
    fn clus(self, i: Int) -> Int32:
        return self.clus_[i]


struct SiPixelDigisCUDA(Movable):
    # Consumed by downstream device code.
    var xx_d: DeviceUniquePtr[UInt16]
    var yy_d: DeviceUniquePtr[UInt16]
    var adc_d: DeviceUniquePtr[UInt16]
    var moduleInd_d: DeviceUniquePtr[UInt16]
    var clus_d: DeviceUniquePtr[Int32]
    var view_d: DeviceUniquePtr[DeviceConstView]

    # For CPU output.
    var pdigi_d: DeviceUniquePtr[UInt32]
    var rawIdArr_d: DeviceUniquePtr[UInt32]

    var nModules_h: UInt32
    var nDigis_h: UInt32

    fn __init__(out self):
        self.xx_d = DeviceUniquePtr[UInt16]()
        self.yy_d = DeviceUniquePtr[UInt16]()
        self.adc_d = DeviceUniquePtr[UInt16]()
        self.moduleInd_d = DeviceUniquePtr[UInt16]()
        self.clus_d = DeviceUniquePtr[Int32]()
        self.view_d = DeviceUniquePtr[DeviceConstView]()
        self.pdigi_d = DeviceUniquePtr[UInt32]()
        self.rawIdArr_d = DeviceUniquePtr[UInt32]()
        self.nModules_h = 0
        self.nDigis_h = 0

    fn __init__(
        out self,
        maxFedWords: UInt,
        mut dev_state: _AllocateDeviceState,
        mut host_state: _AllocateHostState,
        stream: CUDAStreamType = cudaStreamDefault,
    ) raises:
        self.xx_d = make_device_unique[UInt16](maxFedWords, dev_state, stream)
        self.yy_d = make_device_unique[UInt16](maxFedWords, dev_state, stream)
        self.adc_d = make_device_unique[UInt16](maxFedWords, dev_state, stream)
        self.moduleInd_d = make_device_unique[UInt16](maxFedWords, dev_state, stream)
        self.clus_d = make_device_unique[Int32](maxFedWords, dev_state, stream)
        self.view_d = make_device_unique[DeviceConstView](dev_state, stream)
        self.pdigi_d = make_device_unique[UInt32](maxFedWords, dev_state, stream)
        self.rawIdArr_d = make_device_unique[UInt32](maxFedWords, dev_state, stream)
        self.nModules_h = 0
        self.nDigis_h = 0

        var view = make_host_unique[DeviceConstView](host_state, stream)
        view[].get()[0].xx_ = self.xx_d.get()
        view[].get()[0].yy_ = self.yy_d.get()
        view[].get()[0].adc_ = self.adc_d.get()
        view[].get()[0].moduleInd_ = self.moduleInd_d.get()
        view[].get()[0].clus_ = self.clus_d.get()

        copyAsyncOwned[DeviceConstView](self.view_d, view^, stream)

    fn __moveinit__(out self, deinit take: Self):
        self.xx_d = take.xx_d^
        self.yy_d = take.yy_d^
        self.adc_d = take.adc_d^
        self.moduleInd_d = take.moduleInd_d^
        self.clus_d = take.clus_d^
        self.view_d = take.view_d^
        self.pdigi_d = take.pdigi_d^
        self.rawIdArr_d = take.rawIdArr_d^
        self.nModules_h = take.nModules_h
        self.nDigis_h = take.nDigis_h

    fn setNModulesDigis(mut self, nModules: UInt32, nDigis: UInt32):
        self.nModules_h = nModules
        self.nDigis_h = nDigis

    fn nModules(self) -> UInt32:
        return self.nModules_h

    fn nDigis(self) -> UInt32:
        return self.nDigis_h

    fn xx(self) -> UnsafePointer[UInt16, MutAnyOrigin]:
        return self.xx_d.get()

    fn yy(self) -> UnsafePointer[UInt16, MutAnyOrigin]:
        return self.yy_d.get()

    fn adc(self) -> UnsafePointer[UInt16, MutAnyOrigin]:
        return self.adc_d.get()

    fn moduleInd(self) -> UnsafePointer[UInt16, MutAnyOrigin]:
        return self.moduleInd_d.get()

    fn clus(self) -> UnsafePointer[Int32, MutAnyOrigin]:
        return self.clus_d.get()

    fn pdigi(self) -> UnsafePointer[UInt32, MutAnyOrigin]:
        return self.pdigi_d.get()

    fn rawIdArr(self) -> UnsafePointer[UInt32, MutAnyOrigin]:
        return self.rawIdArr_d.get()

    fn view(self) -> UnsafePointer[DeviceConstView, MutAnyOrigin]:
        return self.view_d.get()

    fn adcToHostAsync(
        self, mut state: _AllocateHostState, stream: CUDAStreamType
    ) raises -> HostUniquePtr[UInt16]:
        var ret = make_host_unique[UInt16](UInt(self.nDigis()), state, stream)
        copyAsync[UInt16](ret, self.adc_d, UInt(self.nDigis()), stream)
        return ret^

    fn clusToHostAsync(
        self, mut state: _AllocateHostState, stream: CUDAStreamType
    ) raises -> HostUniquePtr[Int32]:
        var ret = make_host_unique[Int32](UInt(self.nDigis()), state, stream)
        copyAsync[Int32](ret, self.clus_d, UInt(self.nDigis()), stream)
        return ret^

    fn pdigiToHostAsync(
        self, mut state: _AllocateHostState, stream: CUDAStreamType
    ) raises -> HostUniquePtr[UInt32]:
        var ret = make_host_unique[UInt32](UInt(self.nDigis()), state, stream)
        copyAsync[UInt32](ret, self.pdigi_d, UInt(self.nDigis()), stream)
        return ret^

    fn rawIdArrToHostAsync(
        self, mut state: _AllocateHostState, stream: CUDAStreamType
    ) raises -> HostUniquePtr[UInt32]:
        var ret = make_host_unique[UInt32](UInt(self.nDigis()), state, stream)
        copyAsync[UInt32](ret, self.rawIdArr_d, UInt(self.nDigis()), stream)
        return ret^
