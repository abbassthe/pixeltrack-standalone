# Mojo port of plugin-SiPixelClusterizer/SiPixelClusterThresholds.h. New in
# cudadev -- mojo-serial has no equivalent, it hardcodes a moduleId-based
# charge cut (GPUClustering.mojo:347) instead of this layer-based lookup.
from builtin.device_passable import DevicePassable


@fieldwise_init
struct SiPixelClusterThresholds(
    Copyable, Movable, TrivialRegisterPassable, DevicePassable
):
    var layer1: Int32
    var otherLayers: Int32

    # The C++ kernel receives this two-Int32 POD by value. Declare the same
    # device representation explicitly so CUDA's host launcher can marshal it
    # as one kernel argument rather than splitting the interface into scalars.
    alias device_type: AnyType = Self

    fn _to_device_type[o: MutOrigin, //](self, target: UnsafePointer[NoneType, o]):
        target.bitcast[Self.device_type]()[] = self

    @staticmethod
    fn get_type_name() -> String:
        return "SiPixelClusterThresholds"

    @staticmethod
    fn get_device_type_name() -> String:
        return Self.get_type_name()

    fn getThresholdForLayerOnCondition(self, isLayer1: Bool) -> Int32:
        return self.layer1 if isLayer1 else self.otherLayers


comptime kSiPixelClusterThresholdsDefaultPhase1 = SiPixelClusterThresholds(2000, 4000)
