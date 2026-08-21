# Mojo port of plugin-SiPixelClusterizer/SiPixelROCsStatusAndMappingWrapperESProducer.cc.
#
# cablingMap.bin's struct isn't read via read_obj[SiPixelROCsStatusAndMapping]
# -- confirmed via bisection that read_obj's return-boundary move of this
# specific 1.4MB, InlineArray-heavy struct makes `mojo build` hang (580s+,
# not just slow). Default-constructing the value and memcpy'ing the file's
# raw bytes over it compiles in ~2s and produces byte-identical results
# (cross-checked: both give size=53376 on the real data file). See
# doc/MojoCudaDevPort.md for the full investigation.
from pathlib import Path
from memory import memcpy
from std.sys.info import size_of

from MojoCudaDev.CondFormats.SiPixelFedIds import SiPixelFedIds
from MojoCudaDev.CondFormats.SiPixelROCsStatusAndMapping import SiPixelROCsStatusAndMapping
from MojoCudaDev.CondFormats.SiPixelROCsStatusAndMappingWrapper import (
    SiPixelROCsStatusAndMappingWrapper,
)
from MojoCudaDev.CUDACore.CUDAAppContext import CUDAAppContext
from MojoCudaDev.CUDACore.CUDACompat import cudaStreamDefault
from MojoCudaDev.Framework.ESProducer import ESProducer
from MojoCudaDev.Framework.EventSetup import EventSetup
from MojoCudaDev.MojoBridge.DTypes import Typeable
from MojoCudaDev.MojoBridge.File import read_simd


struct SiPixelROCsStatusAndMappingWrapperESProducer(ESProducer, Typeable):
    var _data: Path

    fn __init__(out self):
        self._data = Path("")

    fn __init__(out self, var path: Path):
        self._data = path^

    fn produce(mut self, mut eventSetup: EventSetup, ctx: UnsafePointer[CUDAAppContext, MutAnyOrigin]):
        try:
            with open(self._data / "fedIds.bin", "r") as file:
                var nfeds = read_simd[DType.uint32](file)
                var fedIds = List[UInt32]()
                for _ in range(Int(nfeds)):
                    fedIds.append(read_simd[DType.uint32](file))
                eventSetup.put[SiPixelFedIds](SiPixelFedIds(fedIds^))
        except e:
            print("Error during loading data in SiPixelROCsStatusAndMappingWrapperESProducer (fedIds.bin):", e)

        try:
            with open(self._data / "cablingMap.bin", "r") as file:
                var obj = SiPixelROCsStatusAndMapping()
                var raw = file.read_bytes(size_of[SiPixelROCsStatusAndMapping]())
                memcpy(
                    dest=UnsafePointer(to=obj).bitcast[UInt8](),
                    src=raw.steal_data().bitcast[UInt8](),
                    count=size_of[SiPixelROCsStatusAndMapping](),
                )
                var modToUnpDefSize = read_simd[DType.uint32](file)
                var modToUnpDefault = file.read_bytes(Int(modToUnpDefSize))
                eventSetup.put[SiPixelROCsStatusAndMappingWrapper](
                    SiPixelROCsStatusAndMappingWrapper(
                        obj,
                        modToUnpDefault^,
                        ctx[].host_state,
                        ctx[].event_cache,
                        ctx[].runtime,
                        cudaStreamDefault,
                    )
                )
        except e:
            print("Error during loading data in SiPixelROCsStatusAndMappingWrapperESProducer (cablingMap.bin):", e)

    @staticmethod
    fn dtype() -> String:
        return "SiPixelROCsStatusAndMappingWrapperESProducer"
