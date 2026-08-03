from memory import bitcast

alias CUresult = Int32

alias CUDA_SUCCESS: CUresult = 0

alias cudaError_t = Int32

# Basic data aliases used by framework and dataformat modules.
alias SizeType = UInt32  # size_t
alias Short = Int16
alias Float = Float32
alias Double = Float64
alias Char = Int8
alias UChar = UInt8

alias cudaSuccess: cudaError_t = 0
alias cudaErrorMemoryAllocation: cudaError_t = 2
alias cudaErrorNotReady: cudaError_t = 600
alias cudaEventDisableTiming: UInt32 = 2
alias cudaHostAllocDefault: UInt32 = 0x00
alias cudaHostAllocWriteCombined: UInt32 = 0x04


trait Typeable:
    @always_inline
    @staticmethod
    fn dtype() -> String:
        ...


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


@fieldwise_init
struct TypeableInt(Copyable, Movable, Typeable, TrivialRegisterPassable):
    var val: Int

    @always_inline
    @staticmethod
    fn dtype() -> String:
        return "TypeableInt"


@fieldwise_init
struct TypeableUInt(Copyable, Movable, Typeable, TrivialRegisterPassable):
    var val: UInt

    @always_inline
    @staticmethod
    fn dtype() -> String:
        return "TypeableUInt"


# Carried over from mojo-serial's MojoBridge/DTypes.mojo. Not present in the
# mojocudatest version, but SOARotation and the CPE conditions rely on it, so
# it is added here rather than reverting the file (see plan section 6).
fn hex_to_float[fld: Int32]() -> Float:
    return bitcast[src_dtype = DType.int32, src_width=1, DType.float32](fld)
