# Mojo port of CondFormats/SiPixelGainCalibrationForHLTGPU.{h,cc}. mojo-serial's
# version is CPU-only (getCPUProduct(), no device machinery); the GPU side
# (ESProduct, getGPUProductAsync) is added fresh here, matching the pattern
# established by SiPixelROCsStatusAndMappingWrapper.mojo. gain is borrowed and
# memcpy'd, not moved, for the same reason established there.
#
# gainForHLTonHost_'s v_pedestals_ isn't meaningful for GPU use (it's whatever
# the caller's gain argument had at construction time) -- the real device
# address only exists once gainDataOnGPU is itself allocated, inside the
# transfer callback. So getGPUProductAsync copies the whole host struct to
# device first (every field right except v_pedestals_), then patches just
# that one field with a second, separate copy -- mirrors .cc:29-35 exactly.
#
# copyAsync's raw-pointer overloads run a GPU kernel that dereferences both
# src and dst from device code (see copyAsync.mojo's header comment) -- so any
# host-side source must be pinned memory, never a plain List or a host-stack
# local (confirmed via CUDA_ERROR_ILLEGAL_ADDRESS crashes in both cases,
# isolated with scratch/full_sequence_spike.mojo). gainData_ is therefore
# HostAllocator-backed, not List-backed, and the v_pedestals_ patch value is
# staged into a pinned single-element HostAllocator rather than a local var.
# That staging buffer is ephemeral (freed when transfer returns), and
# ESProduct.dataForCurrentDeviceAsync does not synchronize after calling
# transfer, so an explicit synchronize is needed before it's dropped --
# mirrors copyAsyncOwned's own internal synchronize-before-free, just inlined
# since no owned overload exists for this raw-pointer shape.
from memory import memcpy
from std.sys.info import size_of

from MojoCudaDev.CondFormats.SiPixelGainForHLTonGPU import (
    SiPixelGainForHLTonGPU,
    SiPixelGainForHLTonGPU_DecodingStructure,
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
    var gainForHLTonGPU: DeviceUniquePtr[SiPixelGainForHLTonGPU]
    var gainDataOnGPU: DeviceUniquePtr[UInt8]

    fn __init__(out self):
        self.gainForHLTonGPU = DeviceUniquePtr[SiPixelGainForHLTonGPU]()
        self.gainDataOnGPU = DeviceUniquePtr[UInt8]()


struct SiPixelGainCalibrationForHLTGPU(Movable, Typeable):
    var gainForHLTonHost_: HostUniquePtr[SiPixelGainForHLTonGPU]
    var gainData_: HostAllocator[UInt8]

    var gpuData_: ESProduct[GPUData]

    fn __init__(
        out self,
        gain: SiPixelGainForHLTonGPU,
        var gainData: List[UInt8],
        mut host_state: _AllocateHostState,
        mut event_cache: EventCache,
        mut runtime: CUDARuntime,
        stream: CUDAStreamType = cudaStreamDefault,
    ) raises:
        self.gainData_ = HostAllocator[UInt8](gainData.__len__())
        memcpy(
            dest=self.gainData_.data(),
            src=gainData.unsafe_ptr(),
            count=gainData.__len__(),
        )
        self.gainForHLTonHost_ = make_host_unique[SiPixelGainForHLTonGPU](host_state, stream)
        memcpy(
            dest=self.gainForHLTonHost_[].get().bitcast[UInt8](),
            src=UnsafePointer(to=gain).bitcast[UInt8](),
            count=size_of[SiPixelGainForHLTonGPU](),
        )
        self.gpuData_ = ESProduct[GPUData](event_cache, runtime)

    fn __moveinit__(out self, deinit take: Self):
        self.gainForHLTonHost_ = take.gainForHLTonHost_^
        self.gainData_ = take.gainData_^
        self.gpuData_ = take.gpuData_^

    fn cpuProduct(self) -> UnsafePointer[SiPixelGainForHLTonGPU, MutAnyOrigin]:
        return self.gainForHLTonHost_[].get()

    fn getGPUProductAsync(
        mut self, mut dev_state: _AllocateDeviceState, stream: CUDAStreamType
    ) raises -> UnsafePointer[SiPixelGainForHLTonGPU, MutAnyOrigin]:
        var dev_state_ptr = UnsafePointer(to=dev_state)
        var gain_host_ptr = UnsafePointer(to=self.gainForHLTonHost_)
        var gain_data_ptr = self.gainData_.data()
        var gain_data_size = UInt(self.gainData_.size())

        fn transfer(mut data: GPUData, s: CUDAStreamType) raises:
            data.gainDataOnGPU = make_device_unique[UInt8](gain_data_size, dev_state_ptr[], s)
            copyAsync[UInt8](data.gainDataOnGPU, gain_data_ptr, gain_data_size, s)

            data.gainForHLTonGPU = make_device_unique[SiPixelGainForHLTonGPU](dev_state_ptr[], s)
            copyAsync[SiPixelGainForHLTonGPU](data.gainForHLTonGPU, gain_host_ptr[], s)

            # patch v_pedestals_ with the just-allocated device data blob's
            # address -- not known until the line above ran. Staged in pinned
            # memory (not a host-stack local) since copyAsync's raw-pointer
            # overloads dereference src from a GPU kernel; synchronized before
            # returning since this staging buffer is freed on return and
            # dataForCurrentDeviceAsync does not synchronize after transfer.
            var device_field_ptr = UnsafePointer(to=data.gainForHLTonGPU.get()[0].v_pedestals_)
            var patch_staging = HostAllocator[
                UnsafePointer[SiPixelGainForHLTonGPU_DecodingStructure, MutAnyOrigin]
            ](1)
            patch_staging[0] = data.gainDataOnGPU.get().bitcast[SiPixelGainForHLTonGPU_DecodingStructure]()
            copyAsync[UnsafePointer[SiPixelGainForHLTonGPU_DecodingStructure, MutAnyOrigin]](
                device_field_ptr, patch_staging.data(), UInt(1), s
            )
            s.synchronize()

        var result = self.gpuData_.dataForCurrentDeviceAsync(stream, transfer)
        return result[].gainForHLTonGPU.get()

    @staticmethod
    fn dtype() -> String:
        return "SiPixelGainCalibrationForHLTGPU"
