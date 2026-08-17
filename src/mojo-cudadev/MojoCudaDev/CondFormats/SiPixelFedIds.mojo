# Mojo port of CondFormats/SiPixelFedIds.h.
from MojoCudaDev.MojoBridge.DTypes import Typeable


struct SiPixelFedIds(Copyable, Movable, Typeable):
    var _fedIds: List[UInt32]

    fn __init__(out self, var fedIds: List[UInt32]):
        self._fedIds = fedIds^

    fn fedIds(self) -> ref [self._fedIds] List[UInt32]:
        return self._fedIds

    @staticmethod
    fn dtype() -> String:
        return "SiPixelFedIds"
