from MojoCudaDev.MojoBridge.DTypes import Typeable


@fieldwise_init
struct SiPixelErrorCompact(
    Copyable, Defaultable, Movable, Typeable, TrivialRegisterPassable
):
    var rawId: UInt32
    var word: UInt32
    var errorType: UInt8
    var fedId: UInt8

    @always_inline
    fn __init__(out self):
        self.rawId = 0
        self.word = 0
        self.errorType = 0
        self.fedId = 0

    @always_inline
    @staticmethod
    fn dtype() -> String:
        return "SiPixelErrorCompact"
