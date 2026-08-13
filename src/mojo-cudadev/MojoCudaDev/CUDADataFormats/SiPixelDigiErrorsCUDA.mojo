# Mojo port of CUDADataFormats/SiPixelDigiErrorsCUDA.{h,cc}. The allocating
# constructor takes explicit allocator-state params C++ doesn't need, matching
# the pattern established by SiPixelDigisCUDA.mojo/SiPixelClustersCUDA.mojo.
# error_h is a persistent field, not an ephemeral local -- copyAsync's
# host->device overload borrows it, exactly as C++'s copyAsync(error_d,
# error_h, stream) does, so it stays valid afterward for
# copyErrorToHostAsync/dataErrorToHostAsync with no extra ceremony.
from MojoCudaDev.CUDACore.device_unique_ptr import (
    unique_ptr as DeviceUniquePtr,
    make_device_unique,
)
from MojoCudaDev.CUDACore.host_unique_ptr import (
    unique_ptr as HostUniquePtr,
    make_host_unique,
    _HostAllocation,
)
from MojoCudaDev.CUDACore.allocate_device import _AllocateDeviceState
from MojoCudaDev.CUDACore.allocate_host import _AllocateHostState
from MojoCudaDev.CUDACore.copyAsync import copyAsync
from MojoCudaDev.CUDACore.memsetAsync import memsetAsync
from MojoCudaDev.CUDACore.SimpleVector import SimpleVector, make_SimpleVector
from MojoCudaDev.CUDACore.CUDACompat import CUDAStreamType, cudaStreamDefault
from MojoCudaDev.DataFormats.SiPixelErrorCompact import SiPixelErrorCompact
from MojoCudaDev.DataFormats.SiPixelFormatterErrors import SiPixelFormatterErrors

comptime SiPixelErrorCompactVector = SimpleVector[
    SiPixelErrorCompact, "SiPixelErrorCompact"
]


fn _null_host_ptr[T: AnyType]() -> HostUniquePtr[T]:
    return HostUniquePtr[T](_HostAllocation[T]())


struct SiPixelDigiErrorsCUDA(Movable):
    var data_d: DeviceUniquePtr[SiPixelErrorCompact]
    var error_d: DeviceUniquePtr[SiPixelErrorCompactVector]
    var error_h: HostUniquePtr[SiPixelErrorCompactVector]
    var formatterErrors_h: SiPixelFormatterErrors

    fn __init__(out self):
        self.data_d = DeviceUniquePtr[SiPixelErrorCompact]()
        self.error_d = DeviceUniquePtr[SiPixelErrorCompactVector]()
        self.error_h = _null_host_ptr[SiPixelErrorCompactVector]()
        self.formatterErrors_h = SiPixelFormatterErrors()

    fn __init__(
        out self,
        maxFedWords: UInt,
        var errors: SiPixelFormatterErrors,
        mut dev_state: _AllocateDeviceState,
        mut host_state: _AllocateHostState,
        stream: CUDAStreamType = cudaStreamDefault,
    ) raises:
        self.data_d = make_device_unique[SiPixelErrorCompact](maxFedWords, dev_state, stream)
        self.error_d = make_device_unique[SiPixelErrorCompactVector](dev_state, stream)
        self.error_h = make_host_unique[SiPixelErrorCompactVector](host_state, stream)
        self.formatterErrors_h = errors^

        memsetAsync[SiPixelErrorCompact](self.data_d, 0, maxFedWords, stream)

        var error_h_ptr = self.error_h[].get()
        _ = make_SimpleVector(error_h_ptr, Int32(maxFedWords), self.data_d.get())
        debug_assert(self.error_h[].get()[0].empty(), "SiPixelDigiErrorsCUDA: error_h must start empty")
        debug_assert(
            Int(self.error_h[].get()[0].capacity()) == Int(maxFedWords),
            "SiPixelDigiErrorsCUDA: error_h capacity must equal maxFedWords",
        )

        copyAsync[SiPixelErrorCompactVector](self.error_d, self.error_h, stream)

    fn __moveinit__(out self, deinit take: Self):
        self.data_d = take.data_d^
        self.error_d = take.error_d^
        self.error_h = take.error_h^
        self.formatterErrors_h = take.formatterErrors_h^

    fn formatterErrors(self) -> ref [self.formatterErrors_h] SiPixelFormatterErrors:
        return self.formatterErrors_h

    fn error(self) -> UnsafePointer[SiPixelErrorCompactVector, MutAnyOrigin]:
        return self.error_d.get()

    fn copyErrorToHostAsync(mut self, stream: CUDAStreamType) raises:
        copyAsync[SiPixelErrorCompactVector](self.error_h, self.error_d, stream)

    fn dataErrorToHostAsync(
        self, mut host_state: _AllocateHostState, stream: CUDAStreamType
    ) raises -> Tuple[SiPixelErrorCompactVector, HostUniquePtr[SiPixelErrorCompact]]:
        var data = make_host_unique[SiPixelErrorCompact](
            UInt(self.error_h[].get()[0].capacity()), host_state, stream
        )
        if not self.error_h[].get()[0].empty():
            copyAsync[SiPixelErrorCompact](
                data, self.data_d, UInt(self.error_h[].get()[0].size()), stream
            )
        var err = self.error_h[].get()[0].copy()
        err.set_data(data[].get())
        return (err^, data^)
