from CUDACompat import CUDAEventType, cudaEventQuery
from MojoBridge.DTypes import cudaError_t, cudaSuccess, cudaErrorNotReady
from cudaCheck import cudaCheck_


fn event_work_has_completed(event: CUDAEventType) -> Bool raises:
    """Returns True if the work captured by the event has completed.
    Returns False if any captured work is still in flight.
    Raises on any other CUDA error.

    Mirrors cms::cuda::eventWorkHasCompleted from CUDACore/eventWorkHasCompleted.h.
    """
    let ret = cudaEventQuery(event)
    if ret == cudaSuccess:
        return True
    if ret == cudaErrorNotReady:
        return False
    _ = cudaCheck_(__source_location().file_name(), __source_location().line(), "cudaEventQuery", ret)
    return False
