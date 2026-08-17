# Mojo port of CondFormats/SiPixelROCsStatusAndMappingWrapper.{h,cc}.
# device_unique_ptr/host_unique_ptr replace C++'s manual cudaMalloc/cudaFree
# in GPUData/ModulesToUnpack (matches this port's usual simplification).
# getGPUProductAsync/getModToUnpAllAsync take explicit _AllocateDeviceState
# params C++ doesn't need, matching this port's established pattern.
#
# cablingMap is borrowed and memcpy'd, not moved -- matches C++ exactly
# (SiPixelROCsStatusAndMappingWrapper.cc:18-25 takes it by const& and does a
# raw std::memcpy, never its own copy/move constructor). Confirmed necessary,
# not just faithful: moving this struct via init_pointee_move (its 6
# InlineArray[UInt32, 57600] + 1 InlineArray[UInt8, 57600] fields) made
# `mojo build` hang past 580s; memcpy compiles in ~2s. See
# doc/MojoCudaDevPort.md for the full investigation.
from memory import memcpy
from std.sys.info import size_of

from MojoCudaDev.CondFormats.SiPixelROCsStatusAndMapping import (
    SiPixelROCsStatusAndMapping,
    MAX_SIZE_BYTE_BOOL,
)
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
from MojoCudaDev.CUDACore.copyAsync import copyAsync
from MojoCudaDev.CUDACore.CUDACompat import CUDAStreamType, cudaStreamDefault, CUDARuntime
from MojoCudaDev.CUDACore.EventCache import EventCache
from MojoCudaDev.CUDACore.ESProduct import ESProduct
from MojoCudaDev.CUDACore.HostAllocator import HostAllocator
from MojoCudaDev.MojoBridge.DTypes import Typeable


struct GPUData(Defaultable, Movable):
    var cablingMapDevice: DeviceUniquePtr[SiPixelROCsStatusAndMapping]

    fn __init__(out self):
        self.cablingMapDevice = DeviceUniquePtr[SiPixelROCsStatusAndMapping]()


struct ModulesToUnpack(Defaultable, Movable):
    var modToUnpDefault: DeviceUniquePtr[UInt8]

    fn __init__(out self):
        self.modToUnpDefault = DeviceUniquePtr[UInt8]()


struct SiPixelROCsStatusAndMappingWrapper(Movable, Typeable):
    var modToUnpDefault: HostAllocator[UInt8]
    var hasQuality_: Bool

    var cablingMapHost: HostUniquePtr[SiPixelROCsStatusAndMapping]

    var gpuData_: ESProduct[GPUData]
    var modToUnp_: ESProduct[ModulesToUnpack]

    fn __init__(
        out self,
        cablingMap: SiPixelROCsStatusAndMapping,
        var modToUnp: List[UInt8],
        mut host_state: _AllocateHostState,
        mut event_cache: EventCache,
        mut runtime: CUDARuntime,
        stream: CUDAStreamType = cudaStreamDefault,
    ) raises:
        self.modToUnpDefault = HostAllocator[UInt8](modToUnp.__len__())
        self.hasQuality_ = True

        self.cablingMapHost = make_host_unique[SiPixelROCsStatusAndMapping](host_state, stream)
        memcpy(
            dest=self.cablingMapHost[].get().bitcast[UInt8](),
            src=UnsafePointer(to=cablingMap).bitcast[UInt8](),
            count=size_of[SiPixelROCsStatusAndMapping](),
        )

        for i in range(modToUnp.__len__()):
            self.modToUnpDefault[i] = modToUnp[i]

        self.gpuData_ = ESProduct[GPUData](event_cache, runtime)
        self.modToUnp_ = ESProduct[ModulesToUnpack](event_cache, runtime)

    fn __moveinit__(out self, deinit take: Self):
        self.modToUnpDefault = take.modToUnpDefault^
        self.hasQuality_ = take.hasQuality_
        self.cablingMapHost = take.cablingMapHost^
        self.gpuData_ = take.gpuData_^
        self.modToUnp_ = take.modToUnp_^

    fn hasQuality(self) -> Bool:
        return self.hasQuality_

    # returns pointer to GPU memory
    fn getGPUProductAsync(
        mut self, mut dev_state: _AllocateDeviceState, stream: CUDAStreamType
    ) raises -> UnsafePointer[SiPixelROCsStatusAndMapping, MutAnyOrigin]:
        var dev_state_ptr = UnsafePointer(to=dev_state)
        var cabling_map_host_ptr = UnsafePointer(to=self.cablingMapHost)

        fn transfer(mut data: GPUData, s: CUDAStreamType) raises:
            data.cablingMapDevice = make_device_unique[SiPixelROCsStatusAndMapping](dev_state_ptr[], s)
            copyAsync[SiPixelROCsStatusAndMapping](data.cablingMapDevice, cabling_map_host_ptr[], s)

        var result = self.gpuData_.dataForCurrentDeviceAsync(stream, transfer)
        return result[].cablingMapDevice.get()

    # returns pointer to GPU memory
    fn getModToUnpAllAsync(
        mut self, mut dev_state: _AllocateDeviceState, stream: CUDAStreamType
    ) raises -> UnsafePointer[UInt8, MutAnyOrigin]:
        var dev_state_ptr = UnsafePointer(to=dev_state)
        var mod_host_ptr = self.modToUnpDefault.data()
        var mod_host_size = UInt(self.modToUnpDefault.size())

        fn transfer(mut data: ModulesToUnpack, s: CUDAStreamType) raises:
            data.modToUnpDefault = make_device_unique[UInt8](UInt(MAX_SIZE_BYTE_BOOL), dev_state_ptr[], s)
            copyAsync[UInt8](data.modToUnpDefault, mod_host_ptr, mod_host_size, s)

        var result = self.modToUnp_.dataForCurrentDeviceAsync(stream, transfer)
        return result[].modToUnpDefault.get()

    @staticmethod
    fn dtype() -> String:
        return "SiPixelROCsStatusAndMappingWrapper"
