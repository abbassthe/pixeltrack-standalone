# Mojo port of plugin-SiPixelClusterizer/ErrorChecker.{h,cc}. Byte-identical
# across serial/cuda/cudadev (confirmed via diff) -- pure rename sweep from
# mojo-serial's own ErrorChecker.mojo, except Errors/DetErrors use
# SiPixelFormatterErrors (this tree's real port of DataFormats/SiPixelFormatterErrors.h,
# the type ErrorChecker.h's typedefs actually describe) rather than mojo-serial's
# separate PixelErrors.mojo/PixelFormatterErrors, which isn't ported here.
#
# C++'s `errors[dummyDetId].push_back(error)` relies on std::map::operator[]
# auto-vivifying a missing key with an empty vector. Mojo's Dict.__getitem__
# raises on a missing key instead -- mojo-serial's port papers over this with
# a try/except that prints and silently drops the error on first occurrence
# per key, which doesn't match C++'s behavior. Here the key is inserted
# explicitly first, matching the real semantics.
from MojoCudaDev.DataFormats.SiPixelRawDataError import SiPixelRawDataError
from MojoCudaDev.DataFormats.SiPixelFormatterErrors import SiPixelFormatterErrors
from MojoCudaDev.DataFormats.FEDHeader import FEDHeader
from MojoCudaDev.DataFormats.FEDTrailer import FEDTrailer
from MojoCudaDev.MojoBridge.DTypes import UChar, Typeable


@fieldwise_init
struct ErrorChecker(Copyable, Defaultable, Movable, Typeable):
    alias Word32 = UInt32
    alias Word64 = UInt64

    alias DetErrors = SiPixelFormatterErrors.V
    alias Errors = SiPixelFormatterErrors

    alias CRC_bits = 1
    alias LINK_bits = 6
    alias ROC_bits = 5
    alias DCOL_bits = 5
    alias PXID_bits = 8
    alias ADC_bits = 8
    alias OMIT_ERR_bits = 1

    alias CRC_shift = 2
    alias ADC_shift = 0
    alias PXID_shift = Self.ADC_shift + Self.ADC_bits
    alias DCOL_shift = Self.PXID_shift + Self.PXID_bits
    alias ROC_shift = Self.DCOL_shift + Self.DCOL_bits
    alias LINK_shift = Self.ROC_shift + Self.ROC_bits
    alias OMIT_ERR_shift = 20

    alias dummyDetId: UInt32 = 0xFFFFFFFF

    alias CRC_mask: Self.Word64 = ~(~Self.Word64(0) << Self.CRC_bits)
    alias ERROR_mask: Self.Word32 = ~(~Self.Word32(0) << Self.ROC_bits)
    alias LINK_mask: Self.Word32 = ~(~Self.Word32(0) << Self.LINK_bits)
    alias ROC_mask: Self.Word32 = ~(~Self.Word32(0) << Self.ROC_bits)
    alias OMIT_ERR_mask: Self.Word32 = ~(~Self.Word32(0) << Self.OMIT_ERR_bits)

    var includeErrors: Bool

    @always_inline
    fn __init__(out self):
        self.includeErrors = False

    fn checkCRC(
        self,
        mut errorsInEvent: Bool,
        var fedId: Int32,
        trailer: UnsafePointer[Self.Word64, MutAnyOrigin],
        mut errors: Self.Errors,
    ) raises -> Bool:
        var CRC_BIT: Int32 = (
            (trailer[] >> Self.CRC_shift) & Self.CRC_mask
        ).cast[DType.int32]()
        if CRC_BIT == 0:
            return True
        errorsInEvent = True
        if self.includeErrors:
            alias errorType = 39
            var error = SiPixelRawDataError(trailer[], errorType, fedId)
            if UInt(Self.dummyDetId) not in errors:
                errors[UInt(Self.dummyDetId)] = Self.DetErrors()
            errors[UInt(Self.dummyDetId)].append(error^)
        return False

    fn checkHeader(
        self,
        mut errorsInEvent: Bool,
        var fedId: Int32,
        header: UnsafePointer[Self.Word64, MutAnyOrigin],
        mut errors: Self.Errors,
    ) raises -> Bool:
        var fedHeader = FEDHeader(header.bitcast[UChar]())
        if not fedHeader.check():
            return False
        if fedHeader.sourceID().cast[DType.int32]() != fedId:
            print(
                (
                    "PixelDataFormatter::interpretRawData, fedHeader.sourceID()"
                    " != fedId"
                ),
                ", sourceID = ",
                fedHeader.sourceID(),
                ", fedId = ",
                fedId,
                ", errorType = 32",
                sep="",
            )
            errorsInEvent = True
            if self.includeErrors:
                alias errorType = 32
                var error = SiPixelRawDataError(header[], errorType, fedId)
                if UInt(Self.dummyDetId) not in errors:
                    errors[UInt(Self.dummyDetId)] = Self.DetErrors()
                errors[UInt(Self.dummyDetId)].append(error^)
        return fedHeader.moreHeaders()

    fn checkTrailer(
        self,
        mut errorsInEvent: Bool,
        fedId: Int32,
        nWords: UInt32,
        trailer: UnsafePointer[Self.Word64, MutAnyOrigin],
        mut errors: Self.Errors,
    ) raises -> Bool:
        var fedTrailer = FEDTrailer(trailer.bitcast[UChar]())
        if not fedTrailer.check():
            if self.includeErrors:
                alias errorType = 33
                var error = SiPixelRawDataError(trailer[], errorType, fedId)
                if UInt(Self.dummyDetId) not in errors:
                    errors[UInt(Self.dummyDetId)] = Self.DetErrors()
                errors[UInt(Self.dummyDetId)].append(error^)
            errorsInEvent = True
            print(
                "fedTrailer.check failed, Fed: ",
                fedId,
                ", errorType = 33",
                sep="",
            )
            return False
        if fedTrailer.fragmentLength() != nWords:
            print(
                "fedTrailer.fragmentLength()!= nWords !! Fed: ",
                fedId,
                ", errorType = 34",
                sep="",
            )
            errorsInEvent = True
            if self.includeErrors:
                alias errorType = 34
                var error = SiPixelRawDataError(trailer[], errorType, fedId)
                if UInt(Self.dummyDetId) not in errors:
                    errors[UInt(Self.dummyDetId)] = Self.DetErrors()
                errors[UInt(Self.dummyDetId)].append(error^)
        return fedTrailer.moreTrailers()

    @always_inline
    @staticmethod
    fn dtype() -> String:
        return "ErrorChecker"
