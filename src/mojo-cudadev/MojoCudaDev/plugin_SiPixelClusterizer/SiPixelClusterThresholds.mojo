# Mojo port of plugin-SiPixelClusterizer/SiPixelClusterThresholds.h. New in
# cudadev -- mojo-serial has no equivalent, it hardcodes a moduleId-based
# charge cut (GPUClustering.mojo:347) instead of this layer-based lookup.
@fieldwise_init
struct SiPixelClusterThresholds(Copyable, Movable, TrivialRegisterPassable):
    var layer1: Int32
    var otherLayers: Int32

    fn getThresholdForLayerOnCondition(self, isLayer1: Bool) -> Int32:
        return self.layer1 if isLayer1 else self.otherLayers


comptime kSiPixelClusterThresholdsDefaultPhase1 = SiPixelClusterThresholds(2000, 4000)
