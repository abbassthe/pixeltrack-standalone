from memory import bitcast

comptime CUresult = Int32

comptime CUDA_SUCCESS: CUresult = 0

comptime cudaError_t = Int32

# Basic data comptimees used by framework and dataformat modules.
comptime SizeType = UInt32  # size_t
comptime Short = Int16
comptime Float = Float32
comptime Double = Float64
comptime Char = Int8
comptime UChar = UInt8

comptime cudaSuccess: cudaError_t = 0
comptime cudaErrorMemoryAllocation: cudaError_t = 2
comptime cudaErrorNotReady: cudaError_t = 600
comptime cudaEventDisableTiming: UInt32 = 2
comptime cudaHostAllocDefault: UInt32 = 0x00
comptime cudaHostAllocWriteCombined: UInt32 = 0x04


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


# Also carried over from mojo-serial: HistoContainer.mojo's bin() needs
# std::make_unsigned<T>::type for a logical (non-sign-extending) right shift.
fn signed_to_unsigned[T: DType]() -> DType:
    @parameter
    if T == DType.int8 or T == DType.uint8:
        return DType.uint8
    elif T == DType.int16 or T == DType.uint16:
        return DType.uint16
    elif T == DType.int32 or T == DType.uint32:
        return DType.uint32
    elif T == DType.int64 or T == DType.uint64:
        return DType.uint64
    elif T == DType.int128 or T == DType.uint128:
        return DType.uint128
    elif T == DType.int256 or T == DType.uint256:
        return DType.uint256
    return DType.invalid
