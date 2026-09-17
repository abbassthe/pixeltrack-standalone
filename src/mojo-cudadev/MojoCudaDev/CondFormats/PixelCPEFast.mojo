# Mojo port of CondFormats/PixelCPEFast.{h,cc}. The CPU-side condition storage
# follows the C++ layout: a dynamic DetParams table and three inline structs.
# copyAsync is implemented as a GPU copy kernel, so GPUData owns separate
# pinned staging buffers for those ordinary host values. GPUData is lazily
# filled once per device by ESProduct.
from memory import memcpy
from pathlib import Path
from std.sys.info import size_of

from MojoCudaDev.CondFormats.PixelCPEforGPU import (
    CommonParams,
    DetParams,
    LayerGeometry,
    AverageGeometry,
    ParamsOnGPU,
)
from MojoCudaDev.CUDACore.CUDACompat import CUDAStreamType, cudaStreamDefault, CUDARuntime
from MojoCudaDev.CUDACore.ESProduct import ESProduct
from MojoCudaDev.CUDACore.EventCache import EventCache
from MojoCudaDev.CUDACore.allocate_device import _AllocateDeviceState
from MojoCudaDev.CUDACore.allocate_host import _AllocateHostState
from MojoCudaDev.CUDACore.copyAsync import copyAsync
from MojoCudaDev.CUDACore.device_unique_ptr import (
    unique_ptr as DeviceUniquePtr,
    make_device_unique,
)
from MojoCudaDev.CUDACore.host_unique_ptr import (
    unique_ptr as HostUniquePtr,
    make_host_unique_uninitialized,
    _HostAllocation,
)
from MojoCudaDev.MojoBridge.DTypes import Typeable
from MojoCudaDev.MojoBridge.File import read_simd, read_obj


# OwnedPointer has no empty constructor. GPUData needs an initially empty
# persistent host-table owner, analogous to C++'s null/default GPUData.
fn _null_host_ptr[T: AnyType]() -> HostUniquePtr[T]:
    return HostUniquePtr[T](_HostAllocation[T]())


struct GPUData(Defaultable, Movable):
    var commonParams: DeviceUniquePtr[CommonParams]
    var detParams: DeviceUniquePtr[DetParams]
    var averageGeometry: DeviceUniquePtr[AverageGeometry]
    var layerGeometry: DeviceUniquePtr[LayerGeometry]
    # These are staging only. Unlike C++ cudaMemcpyAsync(), this port's
    # copyAsync launches a GPU kernel and therefore cannot read normal host
    # memory directly.
    var commonParams_h: HostUniquePtr[CommonParams]
    var detParams_h: HostUniquePtr[DetParams]
    var averageGeometry_h: HostUniquePtr[AverageGeometry]
    var layerGeometry_h: HostUniquePtr[LayerGeometry]
    # C++ GPUData::paramsOnGPU_h. It must outlive the asynchronous copy to
    # paramsOnGPU_d, unlike a stack-local staging value.
    var paramsOnGPU_h: HostUniquePtr[ParamsOnGPU]
    # C++ GPUData::paramsOnGPU_d.
    var paramsOnGPU_d: DeviceUniquePtr[ParamsOnGPU]

    fn __init__(out self):
        self.commonParams = DeviceUniquePtr[CommonParams]()
        self.detParams = DeviceUniquePtr[DetParams]()
        self.averageGeometry = DeviceUniquePtr[AverageGeometry]()
        self.layerGeometry = DeviceUniquePtr[LayerGeometry]()
        self.commonParams_h = _null_host_ptr[CommonParams]()
        self.detParams_h = _null_host_ptr[DetParams]()
        self.averageGeometry_h = _null_host_ptr[AverageGeometry]()
        self.layerGeometry_h = _null_host_ptr[LayerGeometry]()
        self.paramsOnGPU_h = _null_host_ptr[ParamsOnGPU]()
        self.paramsOnGPU_d = DeviceUniquePtr[ParamsOnGPU]()


struct PixelCPEFast(Movable, Typeable):
    # Match PixelCPEFast.h: std::vector<DetParams> plus three value members.
    # The `ParamsOnGPU` CPU product below is just their non-owning pointer
    # table, exactly as in the C++ constructor.
    var m_detParamsGPU: List[DetParams]
    var m_commonParamsGPU: CommonParams
    var m_layerGeometry: LayerGeometry
    var m_averageGeometry: AverageGeometry
    var cpuData_: ParamsOnGPU
    var gpuData_: ESProduct[GPUData]
    var hostState_: UnsafePointer[_AllocateHostState, MutAnyOrigin]

    fn __init__(
        out self,
        var path: Path,
        mut host_state: _AllocateHostState,
        mut event_cache: EventCache,
        mut runtime: CUDARuntime,
        stream: CUDAStreamType = cudaStreamDefault,
    ) raises:
        # Keep deserialization with the owning condition object, matching the
        # C++ constructor. The producer only supplies its data-directory path.
        with open(path, "r") as file:
            self.m_commonParamsGPU = read_obj[CommonParams](file)
            var detParamsCount = Int(read_simd[DType.uint32](file))
            var detParamsBytes = file.read_bytes(
                detParamsCount * size_of[DetParams]()
            )
            self.m_detParamsGPU = List[DetParams](
                unsafe_uninit_length=detParamsCount
            )
            memcpy(
                dest=self.m_detParamsGPU.unsafe_ptr().bitcast[UInt8](),
                src=detParamsBytes.unsafe_ptr(),
                count=detParamsBytes.__len__(),
            )
            self.m_averageGeometry = read_obj[AverageGeometry](file)
            self.m_layerGeometry = read_obj[LayerGeometry](file)

            self.gpuData_ = ESProduct[GPUData](event_cache, runtime)
            self.hostState_ = UnsafePointer(to=host_state)
            self.cpuData_ = ParamsOnGPU()
            self._setCPUProductPointers()

    fn __moveinit__(out self, deinit take: Self):
        self.m_detParamsGPU = take.m_detParamsGPU^
        self.m_commonParamsGPU = take.m_commonParamsGPU^
        self.m_layerGeometry = take.m_layerGeometry^
        self.m_averageGeometry = take.m_averageGeometry^
        self.gpuData_ = take.gpuData_^
        self.hostState_ = take.hostState_
        self.cpuData_ = ParamsOnGPU()
        # Inline members have new addresses after a move, so rebuild the
        # borrowed CPU pointer table. The dynamic DetParams list keeps its
        # storage address as the equivalent C++ vector does.
        self._setCPUProductPointers()

    fn _setCPUProductPointers(mut self):
        self.cpuData_ = ParamsOnGPU()
        self.cpuData_.m_commonParams = UnsafePointer(to=self.m_commonParamsGPU)
        self.cpuData_.m_detParams = self.m_detParamsGPU.unsafe_ptr()
        self.cpuData_.m_layerGeometry = UnsafePointer(to=self.m_layerGeometry)
        self.cpuData_.m_averageGeometry = UnsafePointer(to=self.m_averageGeometry)

    # C++: PixelCPEFast::getCPUProduct(). As with C++ references returned from
    # an owning value, this pointer table is valid while this object remains
    # at its current address.
    fn getCPUProduct(self) -> ParamsOnGPU:
        return self.cpuData_.copy()

    # C++: PixelCPEFast::getGPUProductAsync(). GPUData retains both the
    # host-side pointer table and its device copy, so all copies stay ordered
    # on `stream` without a synchronization here.
    fn getGPUProductAsync(
        mut self,
        mut dev_state: _AllocateDeviceState,
        stream: CUDAStreamType,
    ) raises -> UnsafePointer[ParamsOnGPU, MutAnyOrigin]:
        var dev_state_ptr = UnsafePointer(to=dev_state)
        var common_params = UnsafePointer(to=self.m_commonParamsGPU)
        var det_params = self.m_detParamsGPU.unsafe_ptr()
        var layer_geometry = UnsafePointer(to=self.m_layerGeometry)
        var average_geometry = UnsafePointer(to=self.m_averageGeometry)
        var host_state = self.hostState_
        var det_params_count = UInt(self.m_detParamsGPU.__len__())

        fn transfer(mut data: GPUData, s: CUDAStreamType) raises:
            data.commonParams = make_device_unique[CommonParams](dev_state_ptr[], s)
            data.detParams = make_device_unique[DetParams](det_params_count, dev_state_ptr[], s)
            data.averageGeometry = make_device_unique[AverageGeometry](dev_state_ptr[], s)
            data.layerGeometry = make_device_unique[LayerGeometry](dev_state_ptr[], s)
            data.paramsOnGPU_d = make_device_unique[ParamsOnGPU](dev_state_ptr[], s)

            # Stage CPU-owned values in mapped host allocations before the
            # GPU copy kernels dereference them. This is implementation
            # machinery, not part of PixelCPEFast's CPU storage model.
            data.commonParams_h = make_host_unique_uninitialized[CommonParams](host_state[], s)
            data.detParams_h = make_host_unique_uninitialized[DetParams](det_params_count, host_state[], s)
            data.averageGeometry_h = make_host_unique_uninitialized[AverageGeometry](host_state[], s)
            data.layerGeometry_h = make_host_unique_uninitialized[LayerGeometry](host_state[], s)
            memcpy(
                dest=data.commonParams_h[].get().bitcast[UInt8](),
                src=common_params.bitcast[UInt8](),
                count=size_of[CommonParams](),
            )
            memcpy(
                dest=data.detParams_h[].get().bitcast[UInt8](),
                src=det_params.bitcast[UInt8](),
                count=Int(det_params_count) * size_of[DetParams](),
            )
            memcpy(
                dest=data.averageGeometry_h[].get().bitcast[UInt8](),
                src=average_geometry.bitcast[UInt8](),
                count=size_of[AverageGeometry](),
            )
            memcpy(
                dest=data.layerGeometry_h[].get().bitcast[UInt8](),
                src=layer_geometry.bitcast[UInt8](),
                count=size_of[LayerGeometry](),
            )
            data.paramsOnGPU_h = make_host_unique_uninitialized[ParamsOnGPU](host_state[], s)

            var params = ParamsOnGPU()
            params.m_commonParams = data.commonParams.get()
            params.m_detParams = data.detParams.get()
            params.m_layerGeometry = data.layerGeometry.get()
            params.m_averageGeometry = data.averageGeometry.get()
            data.paramsOnGPU_h[].get().init_pointee_move(params^)

            # Mirror C++: queue the persistent pointer table first, followed
            # by the four pointed-to payloads. A consumer on this stream sees
            # all of them in order, and ESProduct publishes its event after
            # this callback returns.
            copyAsync[ParamsOnGPU](data.paramsOnGPU_d, data.paramsOnGPU_h, s)

            copyAsync[CommonParams](data.commonParams, data.commonParams_h, s)
            copyAsync[AverageGeometry](data.averageGeometry, data.averageGeometry_h, s)
            copyAsync[LayerGeometry](data.layerGeometry, data.layerGeometry_h, s)
            copyAsync[DetParams](data.detParams, data.detParams_h, det_params_count, s)

        var result = self.gpuData_.dataForCurrentDeviceAsync(stream, transfer)
        return result[].paramsOnGPU_d.get()

    @staticmethod
    fn dtype() -> String:
        return "PixelCPEFast"
