from MojoSerial.MojoBridge.DTypes import SizeType, Typeable


@fieldwise_init
struct SiPixelDigisSoA(Copyable, Defaultable, Movable, Sized, Typeable):
    var _pdigi: List[UInt32]
    var _rawIdArr: List[UInt32]
    var _adc: List[UInt16]
    var _clus: List[Int32]

    # default constructor
    @always_inline
    def __init__(out self):
        self._pdigi = List[UInt32]()
        self._rawIdArr = List[UInt32]()
        self._adc = List[UInt16]()
        self._clus = List[Int32]()

    # unsafe constructor for constructing the SoA object from C-style arrays
    def __init__(
        out self,
        var nDigis: SizeType,
        pdigi: UnsafePointer[UInt32],
        rawIdArr: UnsafePointer[UInt32],
        adc: UnsafePointer[UInt16],
        clus: UnsafePointer[Int32],
    ):
        self._pdigi = List[UInt32](unsafe_uninit_length=UInt(nDigis))
        self._rawIdArr = List[UInt32](unsafe_uninit_length=UInt(nDigis))
        self._adc = List[UInt16](unsafe_uninit_length=UInt(nDigis))
        self._clus = List[Int32](unsafe_uninit_length=UInt(nDigis))
        for i in range(UInt(nDigis)):
            (self._pdigi.unsafe_ptr() + i).init_pointee_copy(pdigi[i])
            (self._rawIdArr.unsafe_ptr() + i).init_pointee_copy(rawIdArr[i])
            (self._adc.unsafe_ptr() + i).init_pointee_copy(adc[i])
            (self._clus.unsafe_ptr() + i).init_pointee_copy(clus[i])
        debug_assert(self._pdigi.__len__() == UInt(nDigis))

    @always_inline
    def __len__(self) -> Int:
        return self._pdigi.__len__()

    @always_inline
    def pdigi(self, var i: SizeType) -> UInt32:
        return self._pdigi[i]

    @always_inline
    def rawIdArr(self, var i: SizeType) -> UInt32:
        return self._rawIdArr[i]

    @always_inline
    def adc(self, var i: SizeType) -> UInt16:
        return self._adc[i]

    @always_inline
    def clus(self, var i: SizeType) -> Int32:
        return self._clus[i]

    @always_inline
    def pdigiList(self) -> ref [self._pdigi] List[UInt32]:
        return self._pdigi

    @always_inline
    def rawIdArrList(self) -> ref [self._rawIdArr] List[UInt32]:
        return self._rawIdArr

    @always_inline
    def adcList(self) -> ref [self._adc] List[UInt16]:
        return self._adc

    @always_inline
    def clusList(self) -> ref [self._clus] List[Int32]:
        return self._clus

    @always_inline
    @staticmethod
    def dtype() -> String:
        return "SiPixelDigisSoA"
