from MojoCudaDev.MojoBridge.DTypes import SizeType, Typeable


@fieldwise_init
struct SiPixelDigisSoA(Copyable, Defaultable, Movable, Sized, Typeable):
    """Delivers digi and cluster data from GPU to host.

    The main purpose of this class is to deliver digi and cluster data from an
    EDProducer that transfers the data from GPU to host to an EDProducer that
    converts the SoA to legacy data products. The class is independent of any
    GPU technology, and in principle could be produced by host code, and be
    used for other purposes than conversion-to-legacy as well.
    """

    var _pdigi: List[UInt32]  # packed digi (row, col, adc) of each pixel
    var _rawIdArr: List[UInt32]  # DetId of each pixel
    var _adc: List[UInt16]  # ADC of each pixel
    var _clus: List[Int32]  # cluster id of each pixel

    # default constructor
    @always_inline
    fn __init__(out self):
        self._pdigi = List[UInt32]()
        self._rawIdArr = List[UInt32]()
        self._adc = List[UInt16]()
        self._clus = List[Int32]()

    # unsafe constructor for constructing the SoA object from C-style arrays
    fn __init__(
        out self,
        var nDigis: SizeType,
        pdigi: UnsafePointer[UInt32, MutAnyOrigin],
        rawIdArr: UnsafePointer[UInt32, MutAnyOrigin],
        adc: UnsafePointer[UInt16, MutAnyOrigin],
        clus: UnsafePointer[Int32, MutAnyOrigin],
    ):
        self._pdigi = List[UInt32](unsafe_uninit_length=Int(nDigis))
        self._rawIdArr = List[UInt32](unsafe_uninit_length=Int(nDigis))
        self._adc = List[UInt16](unsafe_uninit_length=Int(nDigis))
        self._clus = List[Int32](unsafe_uninit_length=Int(nDigis))
        for i in range(Int(nDigis)):
            (self._pdigi.unsafe_ptr() + i).init_pointee_copy(pdigi[i])
            (self._rawIdArr.unsafe_ptr() + i).init_pointee_copy(rawIdArr[i])
            (self._adc.unsafe_ptr() + i).init_pointee_copy(adc[i])
            (self._clus.unsafe_ptr() + i).init_pointee_copy(clus[i])
        debug_assert(len(self._pdigi) == Int(nDigis))

    @always_inline
    fn __len__(self) -> Int:
        return self._pdigi.__len__()

    @always_inline
    fn pdigi(self, var i: SizeType) -> UInt32:
        return self._pdigi[i]

    @always_inline
    fn rawIdArr(self, var i: SizeType) -> UInt32:
        return self._rawIdArr[i]

    @always_inline
    fn adc(self, var i: SizeType) -> UInt16:
        return self._adc[i]

    @always_inline
    fn clus(self, var i: SizeType) -> Int32:
        return self._clus[i]

    @always_inline
    fn pdigiList(self) -> ref [self._pdigi] List[UInt32]:
        return self._pdigi

    @always_inline
    fn rawIdArrList(self) -> ref [self._rawIdArr] List[UInt32]:
        return self._rawIdArr

    @always_inline
    fn adcList(self) -> ref [self._adc] List[UInt16]:
        return self._adc

    @always_inline
    fn clusList(self) -> ref [self._clus] List[Int32]:
        return self._clus

    @always_inline
    @staticmethod
    fn dtype() -> String:
        return "SiPixelDigisSoA"
