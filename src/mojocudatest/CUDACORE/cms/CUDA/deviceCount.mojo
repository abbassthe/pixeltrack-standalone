from std.gpu.host import DeviceContext

fn deviceCount() -> Int:
    # If you specifically want the CUDA device count (like cudaGetDeviceCount), you can pass api="cuda"
    try:
        return DeviceContext.number_of_devices()
    except e:
        return 0
