from std.memory import OwnedPointer

from MojoSerial.CUDACore.SimpleVector import SimpleVector, make_SimpleVector
from MojoSerial.DataFormats.PixelErrors import (
    PixelErrorCompact,
    PixelFormatterErrors,
)
from MojoSerial.MojoBridge.DTypes import SizeType, Typeable


struct SiPixelDigiErrorsSoA(Defaultable, Movable, Typeable):
    comptime _error_dtype = SimpleVector[
        PixelErrorCompact, PixelErrorCompact.dtype()
    ]
    var data_d: OwnedPointer[List[PixelErrorCompact]]
    var error_d: OwnedPointer[Self._error_dtype]
    var formatterErrors_h: PixelFormatterErrors

    @always_inline
    def __init__(out self):
        self.data_d = OwnedPointer(List[PixelErrorCompact]())
        self.error_d = OwnedPointer(Self._error_dtype())
        self.formatterErrors_h = PixelFormatterErrors()

    @always_inline
    def __init__(
        out self, maxFedWords: SizeType, var errors: PixelFormatterErrors
    ):
        self.formatterErrors_h = errors^
        self.data_d = OwnedPointer(
            List[PixelErrorCompact](
                length=UInt(maxFedWords), fill=PixelErrorCompact()
            )
        )
        self.error_d = OwnedPointer(
            make_SimpleVector[PixelErrorCompact, PixelErrorCompact.dtype()](
                maxFedWords.cast[DType.int32](), self.data_d[].unsafe_ptr()
            )
        )
        debug_assert(self.error_d[].empty())
        debug_assert(self.error_d[].capacity() == UInt(maxFedWords))

    @always_inline
    def __init__(out self, *, deinit move: Self):
        self.data_d = move.data_d^
        self.error_d = move.error_d^
        self.formatterErrors_h = move.formatterErrors_h^

    def formatterErrors(
        self,
    ) -> ref [self.formatterErrors_h] PixelFormatterErrors:
        return self.formatterErrors_h

    def error[
        origin: Origin, //
    ](ref [origin]self) -> UnsafePointer[
        Self._error_dtype, mut = origin.mut, origin=origin
    ]:
        return self.error_d.unsafe_ptr()

    def c_error(self) -> UnsafePointer[Self._error_dtype, mut=False]:
        return self.error_d.unsafe_ptr()

    @always_inline
    @staticmethod
    def dtype() -> String:
        return "SiPixelDigiErrorsSoA"
