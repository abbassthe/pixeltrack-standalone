from MojoSerial.DataFormats.BeamSpotPOD import BeamSpotPOD
from MojoSerial.CUDADataFormats.SiPixelClustersSoA import SiPixelClustersSoA
from MojoSerial.CUDADataFormats.SiPixelDigisSoA import SiPixelDigisSoA
from MojoSerial.CondFormats.PixelCPEforGPU import ParamsOnGPU
from MojoSerial.MojoBridge.DTypes import Typeable
from MojoSerial.plugin_SiPixelRecHits.GPUPixelRecHits import getHits
from MojoSerial.CUDACore.HistoContainer import fillManyFromVector


import MojoSerial.CUDADataFormats.TrackingRecHit2DHeterogeneous as TrackingRecHit2DHeterogeneous


def setHitsLayerStart(
    hitsModuleStart: UnsafePointer[UInt32],
    cpeParams: UnsafePointer[ParamsOnGPU],
    hitsLayerStart: UnsafePointer[UInt32],
):
    debug_assert(hitsModuleStart[0] == 0)

    for i in range(11):
        hitsLayerStart[i] = hitsModuleStart[
            cpeParams[].layerGeometry().layerStart[i]
        ]


@fieldwise_init
struct PixelRecHitGPUKernel(Defaultable, Typeable):
    def makeHits(
        self,
        ref digis_d: SiPixelDigisSoA,
        ref clusters_d: SiPixelClustersSoA,
        ref bs_d: BeamSpotPOD,
        var cpeParams: UnsafePointer[ParamsOnGPU],
    ) -> TrackingRecHit2DHeterogeneous.TrackingRecHit2DCPU:
        var nHits = clusters_d.nClusters()
        var hits_d = TrackingRecHit2DHeterogeneous.TrackingRecHit2DCPU(
            nHits, cpeParams, clusters_d.clusModuleStart()
        )

        if digis_d.nModules():  # protect from empty events
            getHits(
                cpeParams,
                UnsafePointer(to=bs_d),
                digis_d.view(),
                digis_d.nDigis(),
                clusters_d.view(),
                hits_d.view(),
            )

        # assuming full warp of threads is better than a smaller number...
        if nHits:
            setHitsLayerStart(
                clusters_d.clusModuleStart(), cpeParams, hits_d.hitsLayerStart()
            )

        if nHits:
            fillManyFromVector(
                hits_d.phiBinner()[],
                10,
                hits_d.iphi(),
                hits_d.hitsLayerStart(),
                nHits,
            )

        return hits_d^

    @always_inline
    @staticmethod
    def dtype() -> String:
        return "PixelRecHitGPUKernel"
