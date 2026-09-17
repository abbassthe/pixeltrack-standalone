# Mojo port of plugin-SiPixelClusterizer/SiPixelRawToClusterCUDA.cc.
#
# The C++ module is EDProducerExternalWork. The current Mojo framework has
# only EDProducer (its continuation hook is intentionally unimplemented in
# ScopedContextAcquire), so this producer waits for its one CUDA stream before
# publishing the result. The device work and Product stream/event ownership
# remain asynchronous with respect to later consumers; the wait is the sole
# framework limitation and can be removed when external-work continuations
# are available.
from MojoCudaDev.CUDACore.CUDAAppContext import CUDAAppContext
from MojoCudaDev.CUDACore.Product import Product
from MojoCudaDev.CUDACore.ScopedContext import ScopedContextProduce
from MojoCudaDev.CUDADataFormats.SiPixelClustersCUDA import SiPixelClustersCUDA
from MojoCudaDev.CUDADataFormats.SiPixelDigiErrorsCUDA import SiPixelDigiErrorsCUDA
from MojoCudaDev.CUDADataFormats.SiPixelDigisCUDA import SiPixelDigisCUDA
from MojoCudaDev.CondFormats.SiPixelFedIds import SiPixelFedIds
from MojoCudaDev.CondFormats.SiPixelGainCalibrationForHLTGPU import (
    SiPixelGainCalibrationForHLTGPU,
)
from MojoCudaDev.CondFormats.SiPixelROCsStatusAndMappingWrapper import (
    SiPixelROCsStatusAndMappingWrapper,
)
from MojoCudaDev.DataFormats.FEDNumbering import FEDNumbering
from MojoCudaDev.DataFormats.FEDRawDataCollection import FEDRawDataCollection
from MojoCudaDev.DataFormats.SiPixelFormatterErrors import SiPixelFormatterErrors
from MojoCudaDev.Framework.EDGetToken import EDGetTokenT
from MojoCudaDev.Framework.EDProducer import EDProducer
from MojoCudaDev.Framework.EDPutToken import EDPutTokenT
from MojoCudaDev.Framework.Event import Event
from MojoCudaDev.Framework.EventSetup import EventSetup
from MojoCudaDev.Framework.ProductRegistry import ProductRegistry
from MojoCudaDev.MojoBridge.DTypes import Typeable
from MojoCudaDev.plugin_SiPixelClusterizer.ErrorChecker import ErrorChecker
from MojoCudaDev.plugin_SiPixelClusterizer.SiPixelClusterThresholds import (
    SiPixelClusterThresholds,
    kSiPixelClusterThresholdsDefaultPhase1,
)
from MojoCudaDev.plugin_SiPixelClusterizer.SiPixelRawToClusterGPUKernel import (
    SiPixelRawToClusterGPUKernel,
    WordFedAppender,
)


struct SiPixelRawToClusterCUDA(EDProducer, Typeable):
    var rawGetToken_: EDGetTokenT[FEDRawDataCollection]
    var digiPutToken_: EDPutTokenT[Product[SiPixelDigisCUDA]]
    var digiErrorPutToken_: EDPutTokenT[Product[SiPixelDigiErrorsCUDA]]
    var clusterPutToken_: EDPutTokenT[Product[SiPixelClustersCUDA]]

    var gpuAlgo_: SiPixelRawToClusterGPUKernel
    var wordFedAppender_: WordFedAppender
    var errors_: SiPixelFormatterErrors

    var isRun2_: Bool
    var includeErrors_: Bool
    var useQuality_: Bool
    var clusterThresholds_: SiPixelClusterThresholds

    fn __init__(out self):
        self.rawGetToken_ = EDGetTokenT[FEDRawDataCollection]()
        self.digiPutToken_ = EDPutTokenT[Product[SiPixelDigisCUDA]]()
        self.digiErrorPutToken_ = EDPutTokenT[Product[SiPixelDigiErrorsCUDA]]()
        self.clusterPutToken_ = EDPutTokenT[Product[SiPixelClustersCUDA]]()
        self.gpuAlgo_ = SiPixelRawToClusterGPUKernel()
        self.wordFedAppender_ = WordFedAppender()
        self.errors_ = SiPixelFormatterErrors()
        self.isRun2_ = True
        self.includeErrors_ = True
        self.useQuality_ = True
        self.clusterThresholds_ = kSiPixelClusterThresholdsDefaultPhase1

    fn __init__(out self, mut reg: ProductRegistry):
        self.rawGetToken_ = EDGetTokenT[FEDRawDataCollection]()
        self.digiPutToken_ = EDPutTokenT[Product[SiPixelDigisCUDA]]()
        self.digiErrorPutToken_ = EDPutTokenT[Product[SiPixelDigiErrorsCUDA]]()
        self.clusterPutToken_ = EDPutTokenT[Product[SiPixelClustersCUDA]]()
        self.gpuAlgo_ = SiPixelRawToClusterGPUKernel()
        self.wordFedAppender_ = WordFedAppender()
        self.errors_ = SiPixelFormatterErrors()
        self.isRun2_ = True
        self.includeErrors_ = True
        self.useQuality_ = True
        self.clusterThresholds_ = kSiPixelClusterThresholdsDefaultPhase1
        try:
            self.rawGetToken_ = reg.consumes[FEDRawDataCollection]()
            self.digiPutToken_ = reg.produces[Product[SiPixelDigisCUDA]]()
            self.clusterPutToken_ = reg.produces[Product[SiPixelClustersCUDA]]()
            if self.includeErrors_:
                self.digiErrorPutToken_ = reg.produces[Product[SiPixelDigiErrorsCUDA]]()
        except e:
            print("Error during SiPixelRawToClusterCUDA construction:", e)

    fn produce(
        mut self,
        mut event: Event,
        ref eventSetup: EventSetup,
        ctx: UnsafePointer[CUDAAppContext, MutAnyOrigin],
    ):
        try:
            var sctx = ScopedContextProduce(event.streamID(), ctx)

            var hgpuMap = rebind[
                UnsafePointer[SiPixelROCsStatusAndMappingWrapper, MutAnyOrigin]
            ](UnsafePointer(to=eventSetup.get[SiPixelROCsStatusAndMappingWrapper]()))
            if hgpuMap[].hasQuality() != self.useQuality_:
                raise (
                    "UseQuality of SiPixelRawToClusterCUDA differs from "
                    "SiPixelROCsStatusAndMappingWrapper"
                )
            var gpuMap = hgpuMap[].getGPUProductAsync(
                ctx[].device_state, sctx.stream()
            )
            var gpuModulesToUnpack = hgpuMap[].getModToUnpAllAsync(
                ctx[].device_state, sctx.stream()
            )

            var hgains = rebind[
                UnsafePointer[SiPixelGainCalibrationForHLTGPU, MutAnyOrigin]
            ](UnsafePointer(to=eventSetup.get[SiPixelGainCalibrationForHLTGPU]()))
            var gpuGains = hgains[].getGPUProductAsync(
                ctx[].device_state, sctx.stream()
            )
            ref fedIds = eventSetup.get[SiPixelFedIds]().fedIds()
            ref buffers = event.get(self.rawGetToken_)

            self.errors_ = SiPixelFormatterErrors()
            self.wordFedAppender_.ensureAllocated()
            var wordCounterGPU: UInt32 = 0
            var fedCounter: UInt32 = 0
            var errorsInEvent = False
            var errorcheck = ErrorChecker()

            # This extraction is line-for-line equivalent to the C++ acquire()
            # loop: headers/trailers are stripped before the GPU sees payload
            # words, and corrupt CRC fragments are skipped.
            for fedIdValue in fedIds:
                var fedId = Int32(fedIdValue)
                if fedId == 40:
                    continue
                debug_assert(
                    fedId >= Int32(FEDNumbering.MINSiPixeluTCAFEDID),
                    "SiPixelRawToClusterCUDA: unexpected pixel FED id",
                )
                fedCounter += 1

                ref rawData = buffers.FEDData(Int(fedId))
                var nWords = UInt32(rawData.size() // 8)
                if nWords == 0:
                    continue

                var rawWords = rebind[UnsafePointer[UInt64, MutAnyOrigin]](
                    rawData.data().bitcast[UInt64]()
                )
                var trailerIndex = Int(nWords) - 1
                var trailer = rawWords + trailerIndex
                if not errorcheck.checkCRC(
                    errorsInEvent, fedId, trailer, self.errors_
                ):
                    continue

                var headerIndex = -1
                var header = rawWords
                var moreHeaders = True
                while moreHeaders:
                    headerIndex += 1
                    header = rawWords + headerIndex
                    moreHeaders = errorcheck.checkHeader(
                        errorsInEvent, fedId, header, self.errors_
                    )

                trailerIndex = Int(nWords)
                var moreTrailers = True
                while moreTrailers:
                    trailerIndex -= 1
                    trailer = rawWords + trailerIndex
                    moreTrailers = errorcheck.checkTrailer(
                        errorsInEvent, fedId, nWords, trailer, self.errors_
                    )

                var beginWords = (rawWords + headerIndex + 1).bitcast[UInt32]()
                var payloadWords = UInt32(
                    2 * (trailerIndex - headerIndex - 1)
                )
                debug_assert(
                    payloadWords % 2 == 0,
                    "SiPixelRawToClusterCUDA: odd pixel payload word count",
                )
                self.wordFedAppender_.initializeWordFed(
                    fedId, wordCounterGPU, beginWords, payloadWords
                )
                wordCounterGPU += payloadWords

            # The driver takes ownership of the formatter-error map, as does
            # C++'s rvalue reference. Reinitialize the member first so the
            # producer stays destructible if the launch path raises.
            var formatterErrors = self.errors_^
            self.errors_ = SiPixelFormatterErrors()
            self.gpuAlgo_.makeClustersAsync(
                self.isRun2_,
                self.clusterThresholds_,
                gpuMap,
                gpuModulesToUnpack,
                gpuGains,
                self.wordFedAppender_,
                formatterErrors^,
                wordCounterGPU,
                fedCounter,
                self.useQuality_,
                self.includeErrors_,
                False,
                ctx[].device_state,
                ctx[].host_state,
                sctx.stream(),
            )

            # See module comment: the framework does not yet expose the C++
            # EDProducerExternalWork continuation, so count readback must be
            # complete before getResults() reads it on the host.
            sctx.stream().synchronize()
            var results = self.gpuAlgo_.getResults()
            var digis = results.digis^
            results.digis = SiPixelDigisCUDA()
            var clusters = results.clusters^
            results.clusters = SiPixelClustersCUDA()
            sctx.emplace[SiPixelDigisCUDA](event, self.digiPutToken_, digis^)
            sctx.emplace[SiPixelClustersCUDA](event, self.clusterPutToken_, clusters^)
            if self.includeErrors_:
                var errors = self.gpuAlgo_.getErrors()
                sctx.emplace[SiPixelDigiErrorsCUDA](
                    event, self.digiErrorPutToken_, errors^
                )
        except e:
            print("Error during SiPixelRawToClusterCUDA.produce:", e)

    fn endJob(mut self) raises:
        pass

    @staticmethod
    fn dtype() -> String:
        return "SiPixelRawToClusterCUDA"
