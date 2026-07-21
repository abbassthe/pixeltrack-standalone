from memory import OwnedPointer
from MojoSerial.CUDACore.CUDACompat import CUDAStreamType
from MojoSerial.MojoBridge.DTypes import SizeType, Typeable, TypeableOwnedPointer

# in principle, a heterogenous SoA implementation regardless of device it runs on should use UnsafePointers based on Mojo's intrinsics

alias HeterogeneousSoA = TypeableOwnedPointer
alias HeterogeneousSoAImpl = TypeableOwnedPointer
alias HeterogeneousSoACPU = HeterogeneousSoAImpl


trait Traits:
    # unable to constraint pointers to pointer trait as it currently does not exist
    alias UniquePointer: AnyType


@deprecated(
    "Heterogenous unique pointers should explicitly rely on Mojo standard"
    " pointers. Please remove any usages of this class."
)
struct CPUTraits[T: AnyType](Traits, Typeable):
    alias UniquePointer = UnsafePointer[T]

    @staticmethod
    fn make_unique(x: CUDAStreamType) -> Self.UniquePointer:
        return Self.UniquePointer.alloc(1)

    @staticmethod
    fn make_unique(size: SizeType, x: CUDAStreamType) -> Self.UniquePointer:
        return Self.UniquePointer.alloc(UInt(size))

    @staticmethod
    fn make_host_unique(x: CUDAStreamType) -> Self.UniquePointer:
        return Self.UniquePointer.alloc(1)

    @staticmethod
    fn make_device_unique(x: CUDAStreamType) -> Self.UniquePointer:
        return Self.UniquePointer.alloc(1)

    @staticmethod
    fn make_device_unique(
        size: SizeType, x: CUDAStreamType
    ) -> Self.UniquePointer:
        return Self.UniquePointer.alloc(UInt(size))

    @staticmethod
    @always_inline
    fn dtype() -> String:
        return "CPUTraits"
