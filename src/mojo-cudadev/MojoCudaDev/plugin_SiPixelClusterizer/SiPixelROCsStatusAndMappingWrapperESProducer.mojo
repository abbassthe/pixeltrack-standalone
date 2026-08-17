# Mojo port of plugin-SiPixelClusterizer/SiPixelROCsStatusAndMappingWrapperESProducer.cc.
#
# produce()'s trait signature (mut self, mut eventSetup: EventSetup) has no
# room for external allocator state -- nothing threads GPU allocator state
# into an ESProducer's produce() call anywhere in this port yet, and
# main.mojo's own TODO flags this producer's registration as still pending
# (Phase 4). So this owns its own _AllocateHostState/EventCache/CUDARuntime,
# constructed once in __init__, rather than block on that broader wiring.
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
from MojoCudaDev.CUDACore.allocate_host import _AllocateHostState
from MojoCudaDev.CUDACore.CUDACompat import CUDARuntime, cudaStreamDefault
from MojoCudaDev.CUDACore.EventCache import EventCache
from MojoCudaDev.Framework.ESProducer import ESProducer
from MojoCudaDev.Framework.EventSetup import EventSetup
from MojoCudaDev.MojoBridge.DTypes import Typeable
from MojoCudaDev.MojoBridge.File import read_simd


struct SiPixelROCsStatusAndMappingWrapperESProducer(ESProducer, Typeable):
    var _data: Path
    var _host_state: _AllocateHostState
    var _event_cache: EventCache
    var _runtime: CUDARuntime

    fn __init__(out self):
        self._data = Path("")
        self._host_state = _AllocateHostState()
        self._event_cache = EventCache()
        self._runtime = CUDARuntime()

    fn __init__(out self, var path: Path):
        self._data = path^
        self._host_state = _AllocateHostState()
        self._event_cache = EventCache()
        self._runtime = CUDARuntime()

    fn produce(mut self, mut eventSetup: EventSetup):
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
                        self._host_state,
                        self._event_cache,
                        self._runtime,
                        cudaStreamDefault,
                    )
                )
        except e:
            print("Error during loading data in SiPixelROCsStatusAndMappingWrapperESProducer (cablingMap.bin):", e)

    @staticmethod
    fn dtype() -> String:
        return "SiPixelROCsStatusAndMappingWrapperESProducer"
