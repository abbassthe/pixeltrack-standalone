from gpu.host import DeviceContext

fn deviceCount() -> Int:
    # If you specifically want the CUDA device count (like cudaGetDeviceCount), you can pass api="cuda"
    return DeviceContext.number_of_devices()