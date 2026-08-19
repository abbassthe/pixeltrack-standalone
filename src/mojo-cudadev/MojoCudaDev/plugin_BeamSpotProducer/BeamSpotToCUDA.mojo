# Mojo port of plugin-BeamSpotProducer/BeamSpotToCUDA.cc. Byte-identical to
# cuda.
from MojoCudaDev.CUDACore.CUDAAppContext import CUDAAppContext
from MojoCudaDev.CUDACore.copyAsync import copyAsync
from MojoCudaDev.CUDACore.host_noncached_unique_ptr import (
    unique_ptr as HostNoncachedUniquePtr,
    make_host_noncached_unique,
)
from MojoCudaDev.CUDACore.Product import Product
from MojoCudaDev.CUDACore.ScopedContext import ScopedContextProduce
from MojoCudaDev.CUDADataFormats.BeamSpotCUDA import BeamSpotCUDA
from MojoCudaDev.DataFormats.BeamSpotPOD import BeamSpotPOD
from MojoCudaDev.Framework.EDProducer import EDProducer
from MojoCudaDev.Framework.EDPutToken import EDPutTokenT
from MojoCudaDev.Framework.Event import Event
from MojoCudaDev.Framework.EventSetup import EventSetup
from MojoCudaDev.Framework.ProductRegistry import ProductRegistry
from MojoCudaDev.MojoBridge.DTypes import Typeable, cudaHostAllocWriteCombined


struct BeamSpotToCUDA(EDProducer, Typeable):
    var _bsPutToken: EDPutTokenT[Product[BeamSpotCUDA]]
    var _bsHost: HostNoncachedUniquePtr[BeamSpotPOD]

    fn __init__(out self):
        self._bsPutToken = EDPutTokenT[Product[BeamSpotCUDA]]()
        self._bsHost = HostNoncachedUniquePtr[BeamSpotPOD](UnsafePointer[BeamSpotPOD, MutAnyOrigin]())

    fn __init__(out self, mut reg: ProductRegistry):
        self._bsPutToken = EDPutTokenT[Product[BeamSpotCUDA]]()
        self._bsHost = HostNoncachedUniquePtr[BeamSpotPOD](UnsafePointer[BeamSpotPOD, MutAnyOrigin]())
        try:
            self._bsPutToken = reg.produces[Product[BeamSpotCUDA]]()
            self._bsHost = make_host_noncached_unique[BeamSpotPOD](cudaHostAllocWriteCombined)
        except e:
            print("Error during BeamSpotToCUDA construction:", e)

    fn produce(mut self, mut event: Event, ref eventSetup: EventSetup, ctx: UnsafePointer[CUDAAppContext, MutAnyOrigin]):
        try:
            self._bsHost.get()[0] = eventSetup.get[BeamSpotPOD]()

            var sctx = ScopedContextProduce(event.streamID(), ctx)

            var bsDevice = BeamSpotCUDA(ctx[].device_state, sctx.stream())
            copyAsync[BeamSpotPOD](bsDevice.ptr(), self._bsHost.get(), UInt(1), sctx.stream())

            sctx.emplace[BeamSpotCUDA](event, self._bsPutToken, bsDevice^)
        except e:
            print("Error during BeamSpotToCUDA.produce:", e)

    fn endJob(mut self):
        pass

    @staticmethod
    fn dtype() -> String:
        return "BeamSpotToCUDA"
