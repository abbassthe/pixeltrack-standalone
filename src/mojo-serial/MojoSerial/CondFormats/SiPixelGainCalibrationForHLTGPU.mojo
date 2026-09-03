from std.collections import Span
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
        self._gainForHLTonHost[].v_pedestals = rebind[
            Span[SiPixelGainForHLTonGPU.DecodingStructure, ImmUntrackedOrigin]
        ](
            Span[SiPixelGainForHLTonGPU.DecodingStructure, _](
                unsafe_ptr=self._gainData.unsafe_ptr().bitcast[
                    SiPixelGainForHLTonGPU.DecodingStructure
                ](),
                length=len(self._gainData) // 2,
            )
        )

    @always_inline
    def getCPUProduct(
        self,
    ) -> ref [self._gainForHLTonHost[]] SiPixelGainForHLTonGPU:
        return self._gainForHLTonHost[]

    @always_inline
    @staticmethod
    def dtype() -> String:
        return "SiPixelGainCalibrationForHLTGPU"
