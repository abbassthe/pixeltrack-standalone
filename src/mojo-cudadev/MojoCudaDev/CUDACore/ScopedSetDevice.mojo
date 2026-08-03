from std.reflection import source_location

from MojoCudaDev.CUDACore.CUDACompat import cudaGetDevice, cudaSetDevice
from MojoCudaDev.CUDACore.cudaCheck import cudaCheck_


struct ScopedSetDevice:
    var prevDevice_: Int

    fn __init__(out self, newDevice: Int) raises:
        self.prevDevice_ = -1
        var loc0 = source_location()
        _ = cudaCheck_(
            String(loc0.file_name),
            loc0.line,
            "cudaGetDevice",
            cudaGetDevice(self.prevDevice_),
        )
        var loc1 = source_location()
        _ = cudaCheck_(
            String(loc1.file_name),
            loc1.line,
            "cudaSetDevice",
            cudaSetDevice(newDevice),
        )
        var currentDevice: Int = -1
        var loc2 = source_location()
        _ = cudaCheck_(
            String(loc2.file_name),
            loc2.line,
            "cudaGetDevice",
            cudaGetDevice(currentDevice),
        )
        if currentDevice != newDevice:
            raise (
                "RuntimeError: ScopedSetDevice failed to select device "
                + String(newDevice)
                + " (current device is "
                + String(currentDevice)
                + ")"
            )

    fn __del__(deinit self):
        # Intentionally don't check the return value to avoid
        # exceptions to be thrown. If this call fails, the process is
        # doomed anyway.
        _ = cudaSetDevice(self.prevDevice_)
