# Mojo port of plugin-Validation/CountValidator.cc.
#
# Ported from mojo-serial's CountValidator.mojo plus the serial->cudadev delta:
# the digi and cluster products are now cms::cuda::Product<> wrappers read
# through a ScopedContextProduce built from the digi product (CountValidator.cc:66-70).
# Track, vertex and reference-count products are unwrapped, as in C++.
from MojoCudaDev.Framework.EDProducer import EDProducer
from MojoCudaDev.Framework.ProductRegistry import ProductRegistry
from MojoCudaDev.Framework.Event import Event
from MojoCudaDev.Framework.EventSetup import EventSetup
from MojoCudaDev.Framework.EDGetToken import EDGetTokenT
from MojoCudaDev.CUDACore.CUDAAppContext import CUDAAppContext
from MojoCudaDev.CUDACore.Product import Product
from MojoCudaDev.CUDACore.ScopedContext import ScopedContextProduce
from MojoCudaDev.DataFormats.DigiClusterCount import DigiClusterCount
from MojoCudaDev.DataFormats.TrackCount import TrackCount
from MojoCudaDev.DataFormats.VertexCount import VertexCount
from MojoCudaDev.CUDADataFormats.SiPixelDigisCUDA import SiPixelDigisCUDA
from MojoCudaDev.CUDADataFormats.SiPixelClustersCUDA import SiPixelClustersCUDA
from MojoCudaDev.CUDADataFormats.PixelTrackHeterogeneous import PixelTrackHeterogeneous
from MojoCudaDev.CUDADataFormats.ZVertexHeterogeneous import ZVertexHeterogeneous

from MojoCudaDev.MojoBridge.DTypes import Typeable


struct CountValidator(Defaultable, EDProducer, Movable, Typeable):
    var digiClusterCountToken_: EDGetTokenT[DigiClusterCount]
    var trackCountToken_: EDGetTokenT[TrackCount]
    var vertexCountToken_: EDGetTokenT[VertexCount]

    var digiToken_: EDGetTokenT[Product[SiPixelDigisCUDA]]
    var clusterToken_: EDGetTokenT[Product[SiPixelClustersCUDA]]
    var trackToken_: EDGetTokenT[PixelTrackHeterogeneous]
    var vertexToken_: EDGetTokenT[ZVertexHeterogeneous]

    # aggregate stats across all events processed by this validator instance.
    # the C++ original keeps these as anonymous-namespace globals (std::atomic);
    # since Mojo has no mutable module-level state ("global vars are not
    # supported"), they live on self instead -- equivalent here since exactly
    # one CountValidator instance processes all events in this framework.
    var allEvents_: UInt32
    var goodEvents_: UInt32
    var sumTrackDifference_: Float32
    var sumVertexDifference_: Int32

    fn __init__(out self):
        self.digiClusterCountToken_ = EDGetTokenT[DigiClusterCount]()
        self.trackCountToken_ = EDGetTokenT[TrackCount]()
        self.vertexCountToken_ = EDGetTokenT[VertexCount]()

        self.digiToken_ = EDGetTokenT[Product[SiPixelDigisCUDA]]()
        self.clusterToken_ = EDGetTokenT[Product[SiPixelClustersCUDA]]()
        self.trackToken_ = EDGetTokenT[PixelTrackHeterogeneous]()
        self.vertexToken_ = EDGetTokenT[ZVertexHeterogeneous]()

        self.allEvents_ = 0
        self.goodEvents_ = 0
        self.sumTrackDifference_ = 0.0
        self.sumVertexDifference_ = 0

    fn __init__(out self, mut reg: ProductRegistry):
        self.allEvents_ = 0
        self.goodEvents_ = 0
        self.sumTrackDifference_ = 0.0
        self.sumVertexDifference_ = 0
        try:
            self.digiClusterCountToken_ = reg.consumes[DigiClusterCount]()
            self.trackCountToken_ = reg.consumes[TrackCount]()
            self.vertexCountToken_ = reg.consumes[VertexCount]()
            self.digiToken_ = reg.consumes[Product[SiPixelDigisCUDA]]()
            self.clusterToken_ = reg.consumes[Product[SiPixelClustersCUDA]]()
            self.trackToken_ = reg.consumes[PixelTrackHeterogeneous]()
            self.vertexToken_ = reg.consumes[ZVertexHeterogeneous]()
        except e:
            print("Handled exception in CountValidator: ", e)
            return Self()

    fn addTrackDifference(mut self, diff: Float32):
        self.sumTrackDifference_ += diff

    fn addVertexDifference(mut self, diff: Int32):
        self.sumVertexDifference_ += diff

    fn incAllEvents(mut self):
        self.allEvents_ += 1

    fn incGoodEvents(mut self):
        self.goodEvents_ += 1

    @staticmethod
    fn strformat[
        Ty: Writable
    ](str: StringSlice, arg: Ty) -> String:
        try:
            return str.format(arg)
        except e:
            return str + " <== Format error: " + String(e)

    @staticmethod
    fn strformat[
        Ty: Writable
    ](str: StringSlice, arg: Ty, arg1: Ty) -> String:
        try:
            return str.format(arg, arg1)
        except e:
            return str + " <== Format error: " + String(e)

    fn produce(
        mut self,
        mut iEvent: Event,
        ref iSetup: EventSetup,
        ctx: UnsafePointer[CUDAAppContext, MutAnyOrigin],
    ):
        try:
            # in 200 runs of 1k events all events are within this tolerance
            var trackTolerance: Float32 = 0.012
            var vertexTolerance: Int32 = 1

            var errorMsg: String = Self.strformat(
                "Event {} ", iEvent.eventID()
            )
            var ok: Bool = True

            ref count = iEvent.get(self.digiClusterCountToken_)

            # CountValidator.cc:66-70 -- the context re-uses the stream that
            # produced the digis, so these reads are ordered after production.
            ref pdigis = iEvent.get(self.digiToken_)
            var sctx = ScopedContextProduce(pdigis.base(), ctx)
            ref digis = sctx.get[SiPixelDigisCUDA](iEvent, self.digiToken_)
            ref clusters = sctx.get[SiPixelClustersCUDA](
                iEvent, self.clusterToken_
            )

            if digis.nModules() != count.nModules():
                errorMsg += Self.strformat(
                    "\n N(modules) is {} expected {}",
                    digis.nModules(),
                    count.nModules(),
                )
                ok = False

            if digis.nDigis() != count.nDigis():
                errorMsg += Self.strformat(
                    "\n N(digis) is {} expected {}",
                    digis.nDigis(),
                    count.nDigis(),
                )
                ok = False
            if clusters.nClusters() != count.nClusters():
                errorMsg += Self.strformat(
                    "\n N(clusters) is {} expected {}",
                    clusters.nClusters(),
                    count.nClusters(),
                )
                ok = False

            ref trackCount = iEvent.get(self.trackCountToken_)
            ref trackWrapper = iEvent.get(self.trackToken_)
            ref tracks = trackWrapper.get()[]

            var nTracks: Int32 = 0
            for i in range(tracks.stride()):
                if tracks.nHits(i) > 0:
                    nTracks += 1

            var rel = abs(
                Float32(nTracks - Int32(trackCount.nTracks()))
                / Float32(trackCount.nTracks())
            )
            if UInt32(nTracks) != trackCount.nTracks():
                self.addTrackDifference(rel)
            if rel >= trackTolerance:
                errorMsg += Self.strformat(
                    "\n N(tracks) is {} expected {}",
                    UInt32(nTracks),
                    trackCount.nTracks(),
                )
                errorMsg += Self.strformat(
                    ", relative difference {} is outside tolerance {}",
                    rel,
                    trackTolerance,
                )
                ok = False

            ref vertexCount = iEvent.get(self.vertexCountToken_)
            ref vertexWrapper = iEvent.get(self.vertexToken_)
            ref vertices = vertexWrapper.get()[]

            var diff: Int32 = abs(
                Int32(vertices.nvFinal) - Int32(vertexCount.nVertices())
            )
            if diff != 0:
                self.addVertexDifference(diff)
            if diff > vertexTolerance:
                errorMsg += Self.strformat(
                    "\n N(vertices) is {} expected {}",
                    vertices.nvFinal,
                    vertexCount.nVertices(),
                )
                errorMsg += Self.strformat(
                    ", difference {} is outside tolerance {}",
                    diff,
                    vertexTolerance,
                )
                ok = False

            self.incAllEvents()
            if ok:
                self.incGoodEvents()
            else:
                print(errorMsg)
        except e:
            print("Error during CountValidator.produce:", e)

    fn endJob(mut self) raises:
        var all = self.allEvents_
        var good = self.goodEvents_
        if all == good:
            print(
                Self.strformat(
                    "CountValidator: all {} events validated successfully!", all
                )
            )
            if self.sumTrackDifference_ > 0:
                print(
                    Self.strformat(
                        (
                            " Average relative track difference: {} (all within"
                            " tolerance)"
                        ),
                        self.sumTrackDifference_ / Float32(all),
                    )
                )
            if self.sumVertexDifference_ > 0:
                print(
                    Self.strformat(
                        (
                            " Average absolute vertex difference: {} (all"
                            " within tolerance)"
                        ),
                        Float32(self.sumVertexDifference_) / Float32(all),
                    )
                )
        else:
            print(
                Self.strformat(
                    (
                        "CountValidator: {} events failed validation (see"
                        " details above)"
                    ),
                    all - good,
                )
            )
            raise Error("CountValidator failed")

    @staticmethod
    fn dtype() -> String:
        return "CountValidator"
