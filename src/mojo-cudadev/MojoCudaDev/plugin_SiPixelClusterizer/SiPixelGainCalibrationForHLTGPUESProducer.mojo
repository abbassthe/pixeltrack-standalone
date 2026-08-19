# Mojo port of plugin-SiPixelClusterizer/SiPixelGainCalibrationForHLTGPUESProducer.cc.
# Byte-identical to cuda. SiPixelGainForHLTonGPU has the same large InlineArray
# field (rangeAndCols_, 2000 elements) as SiPixelROCsStatusAndMapping, so it's
# read via memcpy rather than read_obj for the same reason documented on
# SiPixelROCsStatusAndMappingWrapperESProducer.mojo.
from pathlib import Path
from memory import memcpy
from std.sys.info import size_of

from MojoCudaDev.CondFormats.SiPixelGainForHLTonGPU import SiPixelGainForHLTonGPU
from MojoCudaDev.CondFormats.SiPixelGainCalibrationForHLTGPU import SiPixelGainCalibrationForHLTGPU
from MojoCudaDev.CUDACore.CUDAAppContext import CUDAAppContext
from MojoCudaDev.CUDACore.CUDACompat import cudaStreamDefault
from MojoCudaDev.Framework.ESProducer import ESProducer
from MojoCudaDev.Framework.EventSetup import EventSetup
from MojoCudaDev.MojoBridge.DTypes import Typeable
from MojoCudaDev.MojoBridge.File import read_simd


struct SiPixelGainCalibrationForHLTGPUESProducer(ESProducer, Typeable):
    var _data: Path

    fn __init__(out self):
        self._data = Path("")

    fn __init__(out self, var path: Path):
        self._data = path^

    fn produce(mut self, mut eventSetup: EventSetup, ctx: UnsafePointer[CUDAAppContext, MutAnyOrigin]):
        try:
            with open(self._data / "gain.bin", "r") as file:
                var gain = SiPixelGainForHLTonGPU()
                var raw = file.read_bytes(size_of[SiPixelGainForHLTonGPU]())
                memcpy(
                    dest=UnsafePointer(to=gain).bitcast[UInt8](),
                    src=raw.steal_data().bitcast[UInt8](),
                    count=size_of[SiPixelGainForHLTonGPU](),
                )
                var nbytes = read_simd[DType.uint32](file)
                var gainData = file.read_bytes(Int(nbytes))
                eventSetup.put[SiPixelGainCalibrationForHLTGPU](
                    SiPixelGainCalibrationForHLTGPU(
                        gain,
                        gainData^,
                        ctx[].host_state,
                        ctx[].event_cache,
                        ctx[].runtime,
                        cudaStreamDefault,
                    )
                )
        except e:
            print("Error during loading data in SiPixelGainCalibrationForHLTGPUESProducer:", e)

    @staticmethod
    fn dtype() -> String:
        return "SiPixelGainCalibrationForHLTGPUESProducer"
