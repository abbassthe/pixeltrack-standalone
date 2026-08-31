from MojoSerial.DataFormats.FEDRawData import FEDRawData
from MojoSerial.DataFormats.FEDNumbering import FEDNumbering
from MojoSerial.MojoBridge.DTypes import Typeable


struct FEDRawDataCollection(Copyable, Defaultable, Movable, Typeable):
    var _data: List[FEDRawData]

    @always_inline
    def __init__(out self):
        self._data = List[FEDRawData](
            length=FEDNumbering.lastFEDId() + 1, fill=FEDRawData()
        )

    @always_inline
    def __moveinit__(out self, var other: Self):
        self._data = other._data^

    @always_inline
    def __copyinit__(out self, other: Self):
        self._data = other._data

    @always_inline
    def FEDData(ref self, fedid: Int) -> ref [self._data] FEDRawData:
        return self._data[fedid]

    @always_inline
    def swap(mut self, mut other: Self):
        swap(self._data, other._data)

    @always_inline
    @staticmethod
    def dtype() -> String:
        return "FEDRawDataCollection"


# swap is a library function
