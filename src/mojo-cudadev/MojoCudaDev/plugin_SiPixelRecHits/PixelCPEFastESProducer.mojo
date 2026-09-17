from pathlib import Path

from MojoCudaDev.CondFormats.PixelCPEFast import PixelCPEFast
from MojoCudaDev.CUDACore.CUDAAppContext import CUDAAppContext
from MojoCudaDev.CUDACore.CUDACompat import cudaStreamDefault
from MojoCudaDev.Framework.ESProducer import ESProducer
from MojoCudaDev.Framework.EventSetup import EventSetup
from MojoCudaDev.MojoBridge.DTypes import Typeable


struct PixelCPEFastESProducer(ESProducer, Typeable):
    var _data: Path

    @always_inline
    fn __init__(out self):
        self._data = Path("")

    @always_inline
    fn __init__(out self, var path: Path):
        self._data = path^

    @always_inline
    fn produce(mut self, mut eventSetup: EventSetup, ctx: UnsafePointer[CUDAAppContext, MutAnyOrigin]):
        try:
            eventSetup.put[PixelCPEFast](
                PixelCPEFast(
                    self._data / "cpefast.bin",
                    ctx[].host_state,
                    ctx[].event_cache,
                    ctx[].runtime,
                    cudaStreamDefault,
                )
            )
        except e:
            print(
                "Error during loading data in PixelCPEFastESProducer:",
                e,
            )

    @staticmethod
    fn dtype() -> String:
        return "PixelCPEFastESProducer"
