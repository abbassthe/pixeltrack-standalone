# Mojo port of CondFormats/SiPixelROCsStatusAndMapping.h. C++'s alignas(128)
# on every field forces the whole struct's alignment (and therefore its
# total size) to a multiple of 128 -- confirmed against the real
# data/cablingMap.bin: reading modToUnpDefSize (the field immediately after
# this struct in the file) only lines up at byte offset 1440128, not the
# naive unpadded 1440004. _pad replicates that trailing 124 bytes of
# struct padding; without it, this struct's byte layout wouldn't match the
# file this port reads wholesale via a single binary blob read. Every real
# field already lands on a 128-byte boundary with no inter-field padding
# needed, since MAX_SIZE (57600) is itself a multiple of 128.
comptime MAX_FED: UInt32 = 150
comptime MAX_LINK: UInt32 = 48
comptime MAX_ROC: UInt32 = 8
comptime MAX_SIZE: Int = Int(MAX_FED) * Int(MAX_LINK) * Int(MAX_ROC)
comptime MAX_SIZE_BYTE_BOOL: Int = MAX_SIZE


struct SiPixelROCsStatusAndMapping(Movable):
    var fed: InlineArray[UInt32, MAX_SIZE]
    var link: InlineArray[UInt32, MAX_SIZE]
    var roc: InlineArray[UInt32, MAX_SIZE]
    var rawId: InlineArray[UInt32, MAX_SIZE]
    var rocInDet: InlineArray[UInt32, MAX_SIZE]
    var moduleId: InlineArray[UInt32, MAX_SIZE]
    var badRocs: InlineArray[UInt8, MAX_SIZE]
    var size: UInt32
    var _pad: InlineArray[UInt8, 124]

    fn __init__(out self):
        self.fed = InlineArray[UInt32, MAX_SIZE](fill=0)
        self.link = InlineArray[UInt32, MAX_SIZE](fill=0)
        self.roc = InlineArray[UInt32, MAX_SIZE](fill=0)
        self.rawId = InlineArray[UInt32, MAX_SIZE](fill=0)
        self.rocInDet = InlineArray[UInt32, MAX_SIZE](fill=0)
        self.moduleId = InlineArray[UInt32, MAX_SIZE](fill=0)
        self.badRocs = InlineArray[UInt8, MAX_SIZE](fill=0)
        self.size = 0
        self._pad = InlineArray[UInt8, 124](fill=0)
