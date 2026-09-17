# Focused CudaDev validation for the currently completed pipeline prefix.
# CountValidator also checks tracks/vertices, whose producers are not part of
# this port yet. This validator deliberately stops at rechits: the bundled
# digicluster.bin reference supplies module/digi/cluster counts, and every
# cluster must produce exactly one TrackingRecHit2D entry.
from MojoCudaDev.CUDACore.CUDAAppContext import CUDAAppContext
from MojoCudaDev.CUDACore.Product import Product
from MojoCudaDev.CUDACore.ScopedContext import ScopedContextProduce
from MojoCudaDev.CUDADataFormats.SiPixelClustersCUDA import SiPixelClustersCUDA
from MojoCudaDev.CUDADataFormats.SiPixelDigisCUDA import SiPixelDigisCUDA
from MojoCudaDev.CUDADataFormats.TrackingRecHit2DHeterogeneous import (
    TrackingRecHit2DHeterogeneous,
)
from MojoCudaDev.DataFormats.DigiClusterCount import DigiClusterCount
from MojoCudaDev.Framework.EDGetToken import EDGetTokenT
from MojoCudaDev.Framework.EDProducer import EDProducer
from MojoCudaDev.Framework.Event import Event
from MojoCudaDev.Framework.EventSetup import EventSetup
from MojoCudaDev.Framework.ProductRegistry import ProductRegistry
from MojoCudaDev.MojoBridge.DTypes import Typeable


struct DigiClusterRecHitValidator(EDProducer, Typeable):
    var countToken_: EDGetTokenT[DigiClusterCount]
    var digiToken_: EDGetTokenT[Product[SiPixelDigisCUDA]]
    var clusterToken_: EDGetTokenT[Product[SiPixelClustersCUDA]]
    var hitToken_: EDGetTokenT[Product[TrackingRecHit2DHeterogeneous]]
    var events_: UInt32
    var failures_: UInt32

    fn __init__(out self):
        self.countToken_ = EDGetTokenT[DigiClusterCount]()
        self.digiToken_ = EDGetTokenT[Product[SiPixelDigisCUDA]]()
        self.clusterToken_ = EDGetTokenT[Product[SiPixelClustersCUDA]]()
        self.hitToken_ = EDGetTokenT[Product[TrackingRecHit2DHeterogeneous]]()
        self.events_ = 0
        self.failures_ = 0

    fn __init__(out self, mut reg: ProductRegistry):
        self.countToken_ = EDGetTokenT[DigiClusterCount]()
        self.digiToken_ = EDGetTokenT[Product[SiPixelDigisCUDA]]()
        self.clusterToken_ = EDGetTokenT[Product[SiPixelClustersCUDA]]()
        self.hitToken_ = EDGetTokenT[Product[TrackingRecHit2DHeterogeneous]]()
        self.events_ = 0
        self.failures_ = 0
        try:
            self.countToken_ = reg.consumes[DigiClusterCount]()
            self.digiToken_ = reg.consumes[Product[SiPixelDigisCUDA]]()
            self.clusterToken_ = reg.consumes[Product[SiPixelClustersCUDA]]()
            self.hitToken_ = reg.consumes[
                Product[TrackingRecHit2DHeterogeneous]
            ]()
        except e:
            print("Error during DigiClusterRecHitValidator construction:", e)

    fn produce(
        mut self,
        mut event: Event,
        ref eventSetup: EventSetup,
        ctx: UnsafePointer[CUDAAppContext, MutAnyOrigin],
    ):
        _ = eventSetup
        self.events_ += 1
        try:
            ref reference = event.get(self.countToken_)
            ref pclusters = event.get(self.clusterToken_)
            var sctx = ScopedContextProduce(pclusters.base(), ctx)
            ref digis = sctx.get[SiPixelDigisCUDA](event, self.digiToken_)
            ref clusters = sctx.get[SiPixelClustersCUDA](event, self.clusterToken_)
            ref hits = sctx.get[TrackingRecHit2DHeterogeneous](event, self.hitToken_)

            var ok = (
                digis.nModules() == reference.nModules()
                and digis.nDigis() == reference.nDigis()
                and clusters.nClusters() == reference.nClusters()
                and hits.nHits() == clusters.nClusters()
            )
            if not ok:
                self.failures_ += 1
                print(
                    "DigiClusterRecHitValidator event", event.eventID(),
                    ": got modules/digis/clusters/hits",
                    digis.nModules(), digis.nDigis(), clusters.nClusters(), hits.nHits(),
                    "; expected modules/digis/clusters",
                    reference.nModules(), reference.nDigis(), reference.nClusters(),
                )
        except e:
            self.failures_ += 1
            print("Error during DigiClusterRecHitValidator.produce:", e)

    fn endJob(mut self) raises:
        if self.failures_ == 0:
            print(
                "DigiClusterRecHitValidator: all", self.events_,
                "events validated successfully!",
            )
        else:
            print(
                "DigiClusterRecHitValidator:", self.failures_, "of",
                self.events_, "events failed.",
            )

    @staticmethod
    fn dtype() -> String:
        return "DigiClusterRecHitValidator"
