alias CUresult = Int32

alias CUDA_SUCCESS: CUresult = 0

alias cudaError_t = Int32

alias cudaSuccess: cudaError_t = 0
alias cudaErrorMemoryAllocation: cudaError_t = 2
alias cudaErrorNotReady: cudaError_t = 600
alias cudaEventDisableTiming: UInt32 = 2


@always_inline
fn cuResultName(result: CUresult) -> String:
    if result == CUDA_SUCCESS:
        return "CUDA_SUCCESS"
    return "CUresult(" + String(result) + ")"


@always_inline
fn cuResultMessage(result: CUresult) -> String:
    if result == CUDA_SUCCESS:
        return "no error"
    return "CUDA driver error code " + String(result)


@always_inline
fn cudaErrorName(result: cudaError_t) -> String:
    if result == cudaSuccess:
        return "cudaSuccess"
    return "cudaError_t(" + String(result) + ")"


@always_inline
fn cudaErrorMessage(result: cudaError_t) -> String:
    if result == cudaSuccess:
        return "no error"
    return "CUDA runtime error code " + String(result)
