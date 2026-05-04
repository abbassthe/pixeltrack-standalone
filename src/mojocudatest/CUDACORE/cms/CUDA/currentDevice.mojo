from CUDACompat import cudaGetDevice
from cudaCheck import cudaCheck_


fn currentDevice() raises -> Int:
    var dev: Int = -1
    _ = cudaCheck_(
        __source_location().file_name(),
        __source_location().line(),
        "cudaGetDevice",
        cudaGetDevice(dev),
    )
    return dev
