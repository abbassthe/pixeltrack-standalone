from pathlib import Path

from MojoCudaDev.CUDACore.CUDAAppContext import CUDAAppContext
from MojoCudaDev.Framework.EventSetup import EventSetup


trait ESProducer(Defaultable):
    fn __init__(out self, var path: Path):
        ...

    fn produce(mut self, mut eventSetup: EventSetup, ctx: UnsafePointer[CUDAAppContext, MutAnyOrigin]):
        ...
