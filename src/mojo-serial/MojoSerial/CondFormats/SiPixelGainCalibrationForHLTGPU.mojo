from std.memory import OwnedPointer

from MojoSerial.CondFormats.SiPixelGainForHLTonGPU import SiPixelGainForHLTonGPU
from MojoSerial.MojoBridge.DTypes import Char, Typeable


@fieldwise_init
struct SiPixelGainCalibrationForHLTGPU(Defaultable, Movable, Typeable):
    var _gainForHLTonHost: OwnedPointer[SiPixelGainForHLTonGPU]
    var _gainData: List[Char]

    @always_inline
    def __init__(out self):
        self._gainForHLTonHost = OwnedPointer[SiPixelGainForHLTonGPU](
            SiPixelGainForHLTonGPU()
        )
        self._gainData = []

    @always_inline
    def __init__(
        out self, gain: SiPixelGainForHLTonGPU, var gainData: List[Char]
    ):
        self._gainData = gainData^
        self._gainForHLTonHost = OwnedPointer[SiPixelGainForHLTonGPU](gain)
        self._gainForHLTonHost[].v_pedestals = (
            self._gainData.unsafe_ptr().bitcast[
                SiPixelGainForHLTonGPU.DecodingStructure
            ]()
        )

    @always_inline
    def getCPUProduct(self) -> UnsafePointer[SiPixelGainForHLTonGPU, mut=False]:
        return self._gainForHLTonHost.unsafe_ptr()

    @always_inline
    @staticmethod
    def dtype() -> String:
        return "SiPixelGainCalibrationForHLTGPU"
