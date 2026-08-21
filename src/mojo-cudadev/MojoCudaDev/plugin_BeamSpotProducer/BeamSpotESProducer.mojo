# Mojo port of plugin-BeamSpotProducer/BeamSpotESProducer.cc. Byte-identical
# to cuda.
from pathlib import Path

from MojoCudaDev.DataFormats.BeamSpotPOD import BeamSpotPOD
from MojoCudaDev.Framework.ESProducer import ESProducer
from MojoCudaDev.Framework.EventSetup import EventSetup
from MojoCudaDev.MojoBridge.DTypes import Typeable
from MojoCudaDev.MojoBridge.File import read_obj


struct BeamSpotESProducer(ESProducer, Typeable):
    var _data: Path

    fn __init__(out self):
        self._data = Path("")

    fn __init__(out self, var path: Path):
        self._data = path^

    fn produce(mut self, mut eventSetup: EventSetup):
        try:
            with open(self._data / "beamspot.bin", "r") as file:
                var bs = read_obj[BeamSpotPOD](file)
                eventSetup.put[BeamSpotPOD](bs^)
        except e:
            print("Error during loading data in BeamSpotESProducer:", e)

    @staticmethod
    fn dtype() -> String:
        return "BeamSpotESProducer"
