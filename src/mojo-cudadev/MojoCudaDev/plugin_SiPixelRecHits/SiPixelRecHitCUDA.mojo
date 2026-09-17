# Mojo port of plugin-SiPixelRecHits/SiPixelRecHitCUDA.cc.
from MojoCudaDev.CUDACore.CUDAAppContext import CUDAAppContext
from MojoCudaDev.CUDACore.Product import Product
from MojoCudaDev.CUDACore.ScopedContext import ScopedContextProduce
from MojoCudaDev.CUDADataFormats.BeamSpotCUDA import BeamSpotCUDA
from MojoCudaDev.CUDADataFormats.SiPixelClustersCUDA import SiPixelClustersCUDA
from MojoCudaDev.CUDADataFormats.SiPixelDigisCUDA import SiPixelDigisCUDA
from MojoCudaDev.CUDADataFormats.TrackingRecHit2DHeterogeneous import (
    TrackingRecHit2DHeterogeneous,
)
from MojoCudaDev.CondFormats.PixelCPEFast import PixelCPEFast
from MojoCudaDev.Framework.EDGetToken import EDGetTokenT
from MojoCudaDev.Framework.EDProducer import EDProducer
from MojoCudaDev.Framework.EDPutToken import EDPutTokenT
from MojoCudaDev.Framework.Event import Event
from MojoCudaDev.Framework.EventSetup import EventSetup
from MojoCudaDev.Framework.ProductRegistry import ProductRegistry
from MojoCudaDev.MojoBridge.DTypes import Typeable
from MojoCudaDev.plugin_SiPixelRecHits.PixelRecHits import PixelRecHitGPUKernel


struct SiPixelRecHitCUDA(EDProducer, Typeable):
    var beamSpotToken_: EDGetTokenT[Product[BeamSpotCUDA]]
    var clusterToken_: EDGetTokenT[Product[SiPixelClustersCUDA]]
    var digiToken_: EDGetTokenT[Product[SiPixelDigisCUDA]]
    var hitPutToken_: EDPutTokenT[Product[TrackingRecHit2DHeterogeneous]]
    var gpuAlgo_: PixelRecHitGPUKernel

    fn __init__(out self):
        self.beamSpotToken_ = EDGetTokenT[Product[BeamSpotCUDA]]()
        self.clusterToken_ = EDGetTokenT[Product[SiPixelClustersCUDA]]()
        self.digiToken_ = EDGetTokenT[Product[SiPixelDigisCUDA]]()
        self.hitPutToken_ = EDPutTokenT[Product[TrackingRecHit2DHeterogeneous]]()
        self.gpuAlgo_ = PixelRecHitGPUKernel()

    fn __init__(out self, mut reg: ProductRegistry):
        self.beamSpotToken_ = EDGetTokenT[Product[BeamSpotCUDA]]()
        self.clusterToken_ = EDGetTokenT[Product[SiPixelClustersCUDA]]()
        self.digiToken_ = EDGetTokenT[Product[SiPixelDigisCUDA]]()
        self.hitPutToken_ = EDPutTokenT[Product[TrackingRecHit2DHeterogeneous]]()
        self.gpuAlgo_ = PixelRecHitGPUKernel()
        try:
            self.beamSpotToken_ = reg.consumes[Product[BeamSpotCUDA]]()
            self.clusterToken_ = reg.consumes[Product[SiPixelClustersCUDA]]()
            self.digiToken_ = reg.consumes[Product[SiPixelDigisCUDA]]()
            self.hitPutToken_ = reg.produces[
                Product[TrackingRecHit2DHeterogeneous]
            ]()
        except e:
            print("Error during SiPixelRecHitCUDA construction:", e)

    fn produce(
        mut self,
        mut event: Event,
        ref eventSetup: EventSetup,
        ctx: UnsafePointer[CUDAAppContext, MutAnyOrigin],
    ):
        try:
            # Match C++: the cluster product selects the producing stream;
            # ScopedContext orders all three input products onto it.
            ref pclusters = event.get(self.clusterToken_)
            var sctx = ScopedContextProduce(pclusters.base(), ctx)
            ref clusters = sctx.get[SiPixelClustersCUDA](event, self.clusterToken_)
            ref digis = sctx.get[SiPixelDigisCUDA](event, self.digiToken_)
            ref beamSpot = sctx.get[BeamSpotCUDA](event, self.beamSpotToken_)

            var cpe = rebind[UnsafePointer[PixelCPEFast, MutAnyOrigin]](
                UnsafePointer(to=eventSetup.get[PixelCPEFast]())
            )
            var cpeParams = cpe[].getGPUProductAsync(
                ctx[].device_state, sctx.stream()
            )
            var hits = self.gpuAlgo_.makeHitsAsync(
                digis,
                clusters,
                beamSpot,
                cpeParams,
                ctx[].device_state,
                ctx[].host_state,
                sctx.stream(),
            )
            sctx.emplace[TrackingRecHit2DHeterogeneous](
                event, self.hitPutToken_, hits^
            )
        except e:
            print("Error during SiPixelRecHitCUDA.produce:", e)

    fn endJob(mut self) raises:
        pass

    @staticmethod
    fn dtype() -> String:
        return "SiPixelRecHitCUDA"
