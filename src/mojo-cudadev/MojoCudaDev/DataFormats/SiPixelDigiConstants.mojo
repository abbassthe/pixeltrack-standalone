alias Word64 = UInt64
alias Word32 = UInt32


struct SiPixelConstants:
    alias dummyDetId: UInt32 = 0xFFFFFFFF

    alias CRC_bits: UInt32 = 1
    alias DCOL_bits: UInt32 = 5  # double column
    alias PXID_bits: UInt32 = 8  # pixel id
    alias ADC_bits: UInt32 = 8
    alias OMIT_ERR_bits: UInt32 = 1
    # GO BACK TO OLD VALUES. THE 48-CHAN FED DOES NOT NEED A NEW FORMAT 28/9/16 d.k.
    alias LINK_bits: UInt32 = 6  # 7
    alias ROC_bits: UInt32 = 5  # 4

    alias CRC_shift: UInt32 = 2
    alias ADC_shift: UInt32 = 0
    alias PXID_shift: UInt32 = Self.ADC_shift + Self.ADC_bits
    alias DCOL_shift: UInt32 = Self.PXID_shift + Self.PXID_bits
    alias ROC_shift: UInt32 = Self.DCOL_shift + Self.DCOL_bits
    alias LINK_shift: UInt32 = Self.ROC_shift + Self.ROC_bits
    alias OMIT_ERR_shift: UInt32 = 20

    alias CRC_mask: UInt64 = ~(~Word64(0) << Self.CRC_bits.cast[DType.uint64]())
    alias ERROR_mask: UInt32 = ~(~Word32(0) << Self.ROC_bits)
    alias LINK_mask: UInt32 = ~(~Word32(0) << Self.LINK_bits)
    alias ROC_mask: UInt32 = ~(~Word32(0) << Self.ROC_bits)
    alias OMIT_ERR_mask: UInt32 = ~(~Word32(0) << Self.OMIT_ERR_bits)
    alias DCOL_mask: UInt32 = ~(~Word32(0) << Self.DCOL_bits)
    alias PXID_mask: UInt32 = ~(~Word32(0) << Self.PXID_bits)
    alias ADC_mask: UInt32 = ~(~Word32(0) << Self.ADC_bits)

    # Special for layer 1 bpix rocs 6/9/16 d.k. THIS STAYS.
    alias COL_bits1_l1: UInt32 = 6
    alias ROW_bits1_l1: UInt32 = 7
    alias ROW_shift: UInt32 = Self.ADC_shift + Self.ADC_bits
    alias COL_shift: UInt32 = Self.ROW_shift + Self.ROW_bits1_l1
    alias COL_mask: UInt32 = ~(~Word32(0) << Self.COL_bits1_l1)
    alias ROW_mask: UInt32 = ~(~Word32(0) << Self.ROW_bits1_l1)


@always_inline
fn getLink(ww: UInt32) -> UInt32:
    return (ww >> SiPixelConstants.LINK_shift) & SiPixelConstants.LINK_mask


@always_inline
fn getROC(ww: UInt32) -> UInt32:
    return (ww >> SiPixelConstants.ROC_shift) & SiPixelConstants.ROC_mask


@always_inline
fn getADC(ww: UInt32) -> UInt32:
    return (ww >> SiPixelConstants.ADC_shift) & SiPixelConstants.ADC_mask


@always_inline
fn getCol(ww: UInt32) -> UInt32:
    return (ww >> SiPixelConstants.COL_shift) & SiPixelConstants.COL_mask


@always_inline
fn getRow(ww: UInt32) -> UInt32:
    return (ww >> SiPixelConstants.ROW_shift) & SiPixelConstants.ROW_mask


@always_inline
fn getDCol(ww: UInt32) -> UInt32:
    return (ww >> SiPixelConstants.DCOL_shift) & SiPixelConstants.DCOL_mask


@always_inline
fn getPxId(ww: UInt32) -> UInt32:
    return (ww >> SiPixelConstants.PXID_shift) & SiPixelConstants.PXID_mask
