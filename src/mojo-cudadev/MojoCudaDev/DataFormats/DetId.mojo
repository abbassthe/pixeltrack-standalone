from MojoCudaDev.MojoBridge.DTypes import Typeable


@nonmaterializable(NoneType)
struct Detector:
    alias Tracker: UInt32 = 1
    alias Muon: UInt32 = 2
    alias Ecal: UInt32 = 3
    alias Hcal: UInt32 = 4
    alias Calo: UInt32 = 5
    alias Forward: UInt32 = 6
    alias VeryForward: UInt32 = 7
    alias HGCalEE: UInt32 = 8
    alias HGCalHSi: UInt32 = 9
    alias HGCalHSc: UInt32 = 10
    alias HGCalTrigger: UInt32 = 11


struct DetId(Copyable, Defaultable, Movable, Typeable, TrivialRegisterPassable):
    """Parent class for all detector ids in CMS.

    The DetId is a 32-bit unsigned integer. The four most significant bits
    ([31:28]) identify the large-scale detector (e.g. Tracker or Ecal) while the
    next three bits ([27:25]) identify a part of the detector.
    """

    alias kDetMask: UInt32 = 0xF
    alias kSubdetMask: UInt32 = 0x7
    alias kDetOffset: UInt32 = 28
    alias kSubdetOffset: UInt32 = 25

    var _id: UInt32

    @always_inline
    fn __init__(out self):
        # create an empty or null id (also for persistence)
        self._id = 0

    @always_inline
    fn __init__(out self, id: UInt32):
        # create an id from a raw number
        self._id = id

    @always_inline
    fn __init__(out self, det: UInt32, subdet: Int32):
        # create an id, filling the detector and subdetector fields as specified
        self._id = ((det & Self.kDetMask) << Self.kDetOffset) | (
            (subdet.cast[DType.uint32]() & Self.kSubdetMask)
            << Self.kSubdetOffset
        )

    @always_inline
    fn det(self) -> UInt32:
        # get the detector field from this detid
        return (self._id >> Self.kDetOffset) & Self.kDetMask

    @always_inline
    fn subdetId(self) -> Int32:
        # get the contents of the subdetector field (not cast into any
        # detector's numbering enum)
        var d = self.det()
        if (
            d == Detector.HGCalEE
            or d == Detector.HGCalHSi
            or d == Detector.HGCalHSc
        ):
            return 0
        return ((self._id >> Self.kSubdetOffset) & Self.kSubdetMask).cast[
            DType.int32
        ]()

    @always_inline
    fn rawId(self) -> UInt32:
        # get the raw id
        return self._id

    @always_inline
    fn null(self) -> Bool:
        # is this a null id ?
        return self._id == 0

    @always_inline
    fn __eq__(self, other: Self) -> Bool:
        return self._id == other._id

    @always_inline
    fn __ne__(self, other: Self) -> Bool:
        return self._id != other._id

    @always_inline
    fn __lt__(self, other: Self) -> Bool:
        return self._id < other._id

    @always_inline
    fn __eq__(self, i: UInt32) -> Bool:
        return self._id == i

    @always_inline
    fn __ne__(self, i: UInt32) -> Bool:
        return self._id != i

    @always_inline
    fn __lt__(self, i: UInt32) -> Bool:
        return self._id < i

    @always_inline
    @staticmethod
    fn dtype() -> String:
        return "DetId"
