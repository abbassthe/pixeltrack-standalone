from std.pathlib import Path


from MojoSerial.CondFormats.PixelCPEforGPU import (
    CommonParams,
    DetParams,
    ParamsOnGPU,
    LayerGeometry,
    AverageGeometry,
)
from MojoSerial.MojoBridge.DTypes import Float, Typeable
from MojoSerial.MojoBridge.File import read_simd, read_obj, read_list

comptime micronsToCm: Float = 1.0e-4


struct PixelCPEFast(Defaultable, Movable, Typeable):
    var m_detParamsGPU: List[DetParams]
    var m_commonParamsGPU: CommonParams
    var m_layerGeometry: LayerGeometry
    var m_averageGeometry: AverageGeometry

    var _cpuData: ParamsOnGPU

    @always_inline
    def __init__(out self):
        self.m_detParamsGPU = []
        self.m_commonParamsGPU = CommonParams()
        self.m_layerGeometry = LayerGeometry()
        self.m_averageGeometry = AverageGeometry()

        self._cpuData = ParamsOnGPU(
            UnsafePointer(to=self.m_commonParamsGPU),
            self.m_detParamsGPU.unsafe_ptr(),
            UnsafePointer(to=self.m_layerGeometry),
            UnsafePointer(to=self.m_averageGeometry),
        )

    def __init__(out self, path: Path):
        try:
            with open(path, "r") as file:
                self.m_commonParamsGPU = read_obj[CommonParams](file)
                var ndetParams = read_simd[DType.uint32](file)
                self.m_detParamsGPU = read_list[DetParams](
                    file, Int(ndetParams)
                )
                self.m_averageGeometry = read_obj[AverageGeometry](file)
                self.m_layerGeometry = read_obj[LayerGeometry](file)
        except e:
            print(
                "Error during loading data in PixelCPEFast:",
                e,
            )
            return Self()

        self._cpuData = ParamsOnGPU(
            UnsafePointer(to=self.m_commonParamsGPU),
            self.m_detParamsGPU.unsafe_ptr(),
            UnsafePointer(to=self.m_layerGeometry),
            UnsafePointer(to=self.m_averageGeometry),
        )

    def __moveinit__(out self, var other: Self):
        self.m_detParamsGPU = other.m_detParamsGPU^
        self.m_commonParamsGPU = other.m_commonParamsGPU
        self.m_layerGeometry = other.m_layerGeometry^
        self.m_averageGeometry = other.m_averageGeometry^

        self._cpuData = ParamsOnGPU(
            UnsafePointer(to=self.m_commonParamsGPU),
            self.m_detParamsGPU.unsafe_ptr(),
            UnsafePointer(to=self.m_layerGeometry),
            UnsafePointer(to=self.m_averageGeometry),
        )

    @always_inline
    def getCPUProduct(self) -> ref [self._cpuData] ParamsOnGPU:
        return self._cpuData

    @always_inline
    @staticmethod
    def dtype() -> String:
        return "PixelCPEFast"
