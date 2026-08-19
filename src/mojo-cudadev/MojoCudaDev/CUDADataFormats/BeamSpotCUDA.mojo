# Mojo port of CUDADataFormats/BeamSpotCUDA.h. make_device_unique here takes
# an explicit _AllocateDeviceState param that C++ doesn't need (C++ reaches a
# global allocator implicitly) -- same deviation already used by
# HeterogeneousSoA.toHostAsync.
from MojoCudaDev.CUDACore.device_unique_ptr import unique_ptr as DeviceUniquePtr, make_device_unique
from MojoCudaDev.CUDACore.allocate_device import _AllocateDeviceState
from MojoCudaDev.CUDACore.CUDACompat import CUDAStreamType, cudaStreamDefault
from MojoCudaDev.DataFormats.BeamSpotPOD import BeamSpotPOD
from MojoCudaDev.MojoBridge.DTypes import Typeable


struct BeamSpotCUDA(Defaultable, Movable, Typeable):
    var data_d_: DeviceUniquePtr[BeamSpotPOD]

    fn __init__(out self):
        self.data_d_ = DeviceUniquePtr[BeamSpotPOD]()

    fn __init__(
        out self,
        mut state: _AllocateDeviceState,
        stream: CUDAStreamType = cudaStreamDefault,
    ) raises:
        self.data_d_ = make_device_unique[BeamSpotPOD](state, stream)

    fn __moveinit__(out self, deinit take: Self):
        self.data_d_ = take.data_d_^

    fn data(mut self) -> UnsafePointer[BeamSpotPOD, MutAnyOrigin]:
        return self.data_d_.get()

    fn ptr(ref self) -> ref [self.data_d_] DeviceUniquePtr[BeamSpotPOD]:
        return self.data_d_

    @staticmethod
    fn dtype() -> String:
        return "BeamSpotCUDA"
