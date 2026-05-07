from MojoBridge.DTypes import Typeable, TypeableInt
from CUDACORE.cms.CUDA.CUDAAppContext import CUDAAppContext
from CUDACORE.cms.CUDA.ScopedContext import ScopedContextProduce
from CUDACORE.cms.CUDA.Product import Product
from Framework.EDProducer import EDProducer
from Framework.Event import Event
from Framework.EventSetup import EventSetup
from Framework.EDGetToken import EDGetTokenT
from Framework.EDPutToken import EDPutTokenT
from Framework.ProductRegistry import ProductRegistry
from DataFormats.FEDRawDataCollection import FEDRawDataCollection
from gpuAlgo1 import gpuAlgo1, TypeableFloatBuffer


struct TestProducer(Defaultable, EDProducer, Typeable):
    # heap-allocated so its address is stable for ScopedContext UnsafePointers
    var _cudaCtx: UnsafePointer[CUDAAppContext]
    var _rawGetToken: EDGetTokenT[FEDRawDataCollection]
    var _putToken: EDPutTokenT[Product[TypeableFloatBuffer]]

    @always_inline
    fn __init__(out self):
        self._cudaCtx = UnsafePointer[CUDAAppContext]()
        self._rawGetToken = EDGetTokenT[FEDRawDataCollection]()
        self._putToken = EDPutTokenT[Product[TypeableFloatBuffer]]()

    @always_inline
    fn __init__(out self, mut reg: ProductRegistry):
        self._cudaCtx = UnsafePointer[CUDAAppContext]()
        self._rawGetToken = EDGetTokenT[FEDRawDataCollection]()
        self._putToken = EDPutTokenT[Product[TypeableFloatBuffer]]()
        try:
            self._cudaCtx = UnsafePointer[CUDAAppContext].alloc(1)
            self._cudaCtx.init_pointee_move(CUDAAppContext())
            self._rawGetToken = reg.consumes[FEDRawDataCollection]()
            self._putToken = reg.produces[Product[TypeableFloatBuffer]]()
        except e:
            print("TestProducer init failed,", e)
            if self._cudaCtx != UnsafePointer[CUDAAppContext]():
                self._cudaCtx.destroy_pointee()
                self._cudaCtx.free()
                self._cudaCtx = UnsafePointer[CUDAAppContext]()

    @always_inline
    fn __moveinit__(out self, var other: Self):
        self._cudaCtx = other._cudaCtx
        other._cudaCtx = UnsafePointer[CUDAAppContext]()
        self._rawGetToken = other._rawGetToken
        self._putToken = other._putToken

    fn __del__(var self):
        if self._cudaCtx != UnsafePointer[CUDAAppContext]():
            self._cudaCtx.destroy_pointee()
            self._cudaCtx.free()

    fn produce(mut self, mut event: Event, ref eventSetup: EventSetup):
        if self._cudaCtx == UnsafePointer[CUDAAppContext]():
            print("TestProducer: no CUDA context, skipping event")
            return
        try:
            var value = (
                event.get[FEDRawDataCollection](self._rawGetToken)
                .FEDData(1200)
                .size()
            )
            print(
                "TestProducer  Event ",
                event.eventID(),
                " stream ",
                event.streamID(),
                " ES int ",
                eventSetup.get[TypeableInt]().val,
                " FED 1200 size ",
                value,
                sep="",
            )

            var ctx = ScopedContextProduce(event.streamID(), self._cudaCtx)
            var result = gpuAlgo1(ctx.stream(), self._cudaCtx[])
            ctx.emplace(
                event, self._putToken, TypeableFloatBuffer(result^)
            )
        except e:
            print("Error in TestProducer.produce,", e)

    fn endJob(mut self):
        pass

    @always_inline
    @staticmethod
    fn dtype() -> String:
        return "TestProducer"
