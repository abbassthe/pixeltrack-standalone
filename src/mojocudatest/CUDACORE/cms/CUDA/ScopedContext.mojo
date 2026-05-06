from Framework.EDGetToken import EDGetTokenT
from Framework.EDPutToken import EDPutTokenT
from Framework.Event import Event, StreamID
from Framework.WaitingTaskWithArenaHolder import WaitingTaskWithArenaHolder
from MojoBridge.DTypes import Typeable, cudaEventDisableTiming, cudaSuccess

from CUDACompat import (
    CUDAEventType,
    CUDAStreamType,
    cudaEventCreateWithFlags,
    cudaEventDestroy,
    cudaEventQuery,
    cudaEventRecord,
    cudaSetDevice,
    cudaStreamWaitEvent,
)

from CUDAAppContext import CUDAAppContext
from ContextState import ContextState
from Product import Product
from ProductBase import ProductBase
from SharedEventPtr import SharedEventPtr
from SharedStreamPtr import SharedStreamPtr
from chooseDevice import chooseDevice
from cudaCheck import cudaCheck_


struct ScopedContextBase(Movable):
    var currentDevice_: Int
    var stream_: SharedStreamPtr
    var ctx_: UnsafePointer[CUDAAppContext]

    fn __init__(out self, streamID: StreamID, ctx: UnsafePointer[CUDAAppContext]) raises:
        self.ctx_ = ctx
        self.currentDevice_ = chooseDevice(streamID)
        _ = cudaCheck_(
            __source_location().file_name(),
            __source_location().line(),
            "cudaSetDevice",
            cudaSetDevice(self.currentDevice_),
        )
        self.stream_ = ctx[].stream_cache.get()

    fn __init__(out self, mut data: ProductBase, ctx: UnsafePointer[CUDAAppContext]) raises:
        self.ctx_ = ctx
        self.currentDevice_ = data.device()
        _ = cudaCheck_(
            __source_location().file_name(),
            __source_location().line(),
            "cudaSetDevice",
            cudaSetDevice(self.currentDevice_),
        )

        if data.mayReuseStream():
            var stream = data.streamPtr()
            if not stream:
                raise "RuntimeError: ProductBase stream was missing"
            self.stream_ = stream.value()
        else:
            self.stream_ = ctx[].stream_cache.get()

    fn __init__(out self, device: Int, owned stream: SharedStreamPtr, ctx: UnsafePointer[CUDAAppContext]) raises:
        self.ctx_ = ctx
        self.currentDevice_ = device
        self.stream_ = stream^
        _ = cudaCheck_(
            __source_location().file_name(),
            __source_location().line(),
            "cudaSetDevice",
            cudaSetDevice(self.currentDevice_),
        )

    fn __moveinit__(out self, var other: Self):
        self.currentDevice_ = other.currentDevice_
        self.stream_ = other.stream_^
        self.ctx_ = other.ctx_

    fn device(self) -> Int:
        return self.currentDevice_

    fn stream(self) -> CUDAStreamType:
        return self.stream_[]._item.value()

    fn streamPtr(self) -> SharedStreamPtr:
        return self.stream_


struct ScopedContextGetterBase(Movable):
    var base: ScopedContextBase

    fn __init__(out self, streamID: StreamID, ctx: UnsafePointer[CUDAAppContext]) raises:
        self.base = ScopedContextBase(streamID, ctx)

    fn __init__(out self, mut data: ProductBase, ctx: UnsafePointer[CUDAAppContext]) raises:
        self.base = ScopedContextBase(data, ctx)

    fn __init__(out self, device: Int, owned stream: SharedStreamPtr, ctx: UnsafePointer[CUDAAppContext]) raises:
        self.base = ScopedContextBase(device, stream^, ctx)

    fn __moveinit__(out self, var other: Self):
        self.base = other.base^

    fn device(self) -> Int:
        return self.base.device()

    fn stream(self) -> CUDAStreamType:
        return self.base.stream()

    fn streamPtr(self) -> SharedStreamPtr:
        return self.base.streamPtr()

    fn get[T: Movable & Typeable](mut self, ref data: Product[T]) raises -> ref [data] T:
        self.synchronizeStreams(
            data._base.device(),
            data._base.stream(),
            data._base.isAvailable(),
            data._base.event(),
        )
        return data.data_

    fn get[
        T: Movable & Typeable
    ](mut self, ref event: Event, ref token: EDGetTokenT[Product[T]]) raises -> ref [event] T:
        return self.get[T](event.get[Product[T]](token))

    fn synchronizeStreams(
        mut self,
        dataDevice: Int,
        dataStream: CUDAStreamType,
        available: Bool,
        dataEvent: CUDAEventType,
    ) raises:
        if dataDevice != self.device():
            raise "RuntimeError: Handling data from multiple devices is not yet supported"

        if dataStream != self.stream() and not available:
            _ = cudaCheck_(
                __source_location().file_name(),
                __source_location().line(),
                "cudaStreamWaitEvent",
                cudaStreamWaitEvent(self.stream(), dataEvent, 0),
                "Failed to make a stream wait for an event",
            )


struct ScopedContextHolderHelper(Movable):
    var waitingTaskHolder_: WaitingTaskWithArenaHolder

    fn __init__(out self, var waitingTaskHolder: WaitingTaskWithArenaHolder):
        self.waitingTaskHolder_ = waitingTaskHolder^

    fn __moveinit__(out self, var other: Self):
        self.waitingTaskHolder_ = other.waitingTaskHolder_^

    fn replaceWaitingTaskHolder(mut self, var waitingTaskHolder: WaitingTaskWithArenaHolder):
        self.waitingTaskHolder_ = waitingTaskHolder^

    fn enqueueCallback(mut self, device: Int, stream: CUDAStreamType, ctx: UnsafePointer[CUDAAppContext]):
        _ = device
        var event = CUDAEventType()
        var status = cudaEventCreateWithFlags(event, cudaEventDisableTiming, ctx[].runtime)
        if status != cudaSuccess:
            self.waitingTaskHolder_.doneWaiting(
                "RuntimeError: failed to create completion event for ScopedContext callback"
            )
            return

        status = cudaEventRecord(event, stream)
        if status != cudaSuccess:
            _ = cudaEventDestroy(event)
            self.waitingTaskHolder_.doneWaiting(
                "RuntimeError: failed to record completion event for ScopedContext callback"
            )
            return

        while cudaEventQuery(event) != cudaSuccess:
            pass

        _ = cudaEventDestroy(event)
        self.waitingTaskHolder_.doneWaiting("")

    fn pushNextTask[F: Movable](
        mut self, var func: F, state: UnsafePointer[ContextState]
    ) raises:
        _ = func
        _ = state
        raise (
            "RuntimeError: ScopedContextHolderHelper.pushNextTask() is not available "
            "until WaitingTaskWithArenaHolder can own the task returned by "
            "make_waiting_task_with_holder()"
        )


struct ScopedContextTask(Movable):
    var base: ScopedContextBase
    var holderHelper_: ScopedContextHolderHelper
    var contextState_: UnsafePointer[ContextState]

    fn __init__(
        out self,
        state: UnsafePointer[ContextState],
        var waitingTaskHolder: WaitingTaskWithArenaHolder,
        ctx: UnsafePointer[CUDAAppContext],
    ) raises:
        var stream = state[].streamPtr()
        self.base = ScopedContextBase(state[].device(), stream^, ctx)
        self.holderHelper_ = ScopedContextHolderHelper(waitingTaskHolder^)
        self.contextState_ = state

    fn __moveinit__(out self, var other: Self):
        self.base = other.base^
        self.holderHelper_ = other.holderHelper_^
        self.contextState_ = other.contextState_

    fn __del__(mut self):
        self.holderHelper_.enqueueCallback(self.device(), self.stream(), self.base.ctx_)

    fn device(self) -> Int:
        return self.base.device()

    fn stream(self) -> CUDAStreamType:
        return self.base.stream()

    fn streamPtr(self) -> SharedStreamPtr:
        return self.base.streamPtr()

    fn pushNextTask[F: Movable](mut self, var func: F) raises:
        self.holderHelper_.pushNextTask[F](func^, self.contextState_)

    fn replaceWaitingTaskHolder(mut self, var waitingTaskHolder: WaitingTaskWithArenaHolder):
        self.holderHelper_.replaceWaitingTaskHolder(waitingTaskHolder^)


struct ScopedContextAcquire(Movable):
    var getter: ScopedContextGetterBase
    var holderHelper_: ScopedContextHolderHelper
    var contextState_: UnsafePointer[ContextState]
    var hasContextState_: Bool

    fn __init__(
        out self,
        streamID: StreamID,
        var waitingTaskHolder: WaitingTaskWithArenaHolder,
        ctx: UnsafePointer[CUDAAppContext],
    ) raises:
        self.getter = ScopedContextGetterBase(streamID, ctx)
        self.holderHelper_ = ScopedContextHolderHelper(waitingTaskHolder^)
        self.contextState_ = UnsafePointer[ContextState]()
        self.hasContextState_ = False

    fn __init__(
        out self,
        streamID: StreamID,
        var waitingTaskHolder: WaitingTaskWithArenaHolder,
        mut state: ContextState,
        ctx: UnsafePointer[CUDAAppContext],
    ) raises:
        self.getter = ScopedContextGetterBase(streamID, ctx)
        self.holderHelper_ = ScopedContextHolderHelper(waitingTaskHolder^)
        self.contextState_ = UnsafePointer.address_of(state)
        self.hasContextState_ = True

    fn __init__(
        out self,
        mut data: ProductBase,
        var waitingTaskHolder: WaitingTaskWithArenaHolder,
        ctx: UnsafePointer[CUDAAppContext],
    ) raises:
        self.getter = ScopedContextGetterBase(data, ctx)
        self.holderHelper_ = ScopedContextHolderHelper(waitingTaskHolder^)
        self.contextState_ = UnsafePointer[ContextState]()
        self.hasContextState_ = False

    fn __init__(
        out self,
        mut data: ProductBase,
        var waitingTaskHolder: WaitingTaskWithArenaHolder,
        mut state: ContextState,
        ctx: UnsafePointer[CUDAAppContext],
    ) raises:
        self.getter = ScopedContextGetterBase(data, ctx)
        self.holderHelper_ = ScopedContextHolderHelper(waitingTaskHolder^)
        self.contextState_ = UnsafePointer.address_of(state)
        self.hasContextState_ = True

    fn __moveinit__(out self, var other: Self):
        self.getter = other.getter^
        self.holderHelper_ = other.holderHelper_^
        self.contextState_ = other.contextState_
        self.hasContextState_ = other.hasContextState_
        other.hasContextState_ = False

    fn __del__(mut self):
        self.holderHelper_.enqueueCallback(self.device(), self.stream(), self.getter.base.ctx_)
        if self.hasContextState_:
            try:
                var stream = self.streamPtr()
                self.contextState_[].set(self.device(), stream^)
            except e:
                pass

    fn device(self) -> Int:
        return self.getter.device()

    fn stream(self) -> CUDAStreamType:
        return self.getter.stream()

    fn streamPtr(self) -> SharedStreamPtr:
        return self.getter.streamPtr()

    fn get[T: Movable & Typeable](mut self, ref data: Product[T]) raises -> ref [data] T:
        return self.getter.get[T](data)

    fn get[
        T: Movable & Typeable
    ](mut self, ref event: Event, ref token: EDGetTokenT[Product[T]]) raises -> ref [event] T:
        return self.getter.get[T](event, token)

    fn pushNextTask[F: Movable](mut self, var func: F) raises:
        if not self.hasContextState_:
            self.throwNoState()
        self.holderHelper_.pushNextTask[F](func^, self.contextState_)

    fn replaceWaitingTaskHolder(mut self, var waitingTaskHolder: WaitingTaskWithArenaHolder):
        self.holderHelper_.replaceWaitingTaskHolder(waitingTaskHolder^)

    fn throwNoState(self) raises:
        raise (
            "RuntimeError: Calling ScopedContextAcquire.pushNextTask() requires "
            "ScopedContextAcquire to be constructed with ContextState"
        )


struct ScopedContextProduce(Movable):
    var getter: ScopedContextGetterBase
    var event_: SharedEventPtr

    fn __init__(out self, streamID: StreamID, ctx: UnsafePointer[CUDAAppContext]) raises:
        self.getter = ScopedContextGetterBase(streamID, ctx)
        self.event_ = ctx[].event_cache.get(ctx[].runtime)

    fn __init__(out self, mut data: ProductBase, ctx: UnsafePointer[CUDAAppContext]) raises:
        self.getter = ScopedContextGetterBase(data, ctx)
        self.event_ = ctx[].event_cache.get(ctx[].runtime)

    fn __init__(out self, mut state: ContextState, ctx: UnsafePointer[CUDAAppContext]) raises:
        var stream = state.releaseStreamPtr()
        self.getter = ScopedContextGetterBase(state.device(), stream^, ctx)
        self.event_ = ctx[].event_cache.get(ctx[].runtime)

    fn __init__(
        out self, device: Int, owned stream: SharedStreamPtr, owned event: SharedEventPtr, ctx: UnsafePointer[CUDAAppContext]
    ) raises:
        self.getter = ScopedContextGetterBase(device, stream^, ctx)
        self.event_ = event^

    fn __moveinit__(out self, var other: Self):
        self.getter = other.getter^
        self.event_ = other.event_^

    fn __del__(mut self):
        _ = cudaEventRecord(self.event_[].event, self.stream())

    fn device(self) -> Int:
        return self.getter.device()

    fn stream(self) -> CUDAStreamType:
        return self.getter.stream()

    fn streamPtr(self) -> SharedStreamPtr:
        return self.getter.streamPtr()

    fn get[T: Movable & Typeable](mut self, ref data: Product[T]) raises -> ref [data] T:
        return self.getter.get[T](data)

    fn get[
        T: Movable & Typeable
    ](mut self, ref event: Event, ref token: EDGetTokenT[Product[T]]) raises -> ref [event] T:
        return self.getter.get[T](event, token)

    fn wrap[T: Movable & Typeable](mut self, owned data: T) -> Product[T]:
        var stream = self.streamPtr()
        var event = self.event_.copy()
        return Product[T](self.device(), stream^, event^, data^)

    fn emplace[T: Movable & Typeable](
        mut self, mut event: Event, ref token: EDPutTokenT[Product[T]], owned data: T
    ):
        var stream = self.streamPtr()
        var cuda_event = self.event_.copy()
        var product = Product[T](self.device(), stream^, cuda_event^, data^)
        event.emplace[Product[T]](token, product^)


struct ScopedContextAnalyze(Movable):
    var getter: ScopedContextGetterBase

    fn __init__(out self, mut data: ProductBase) raises:
        self.getter = ScopedContextGetterBase(data)

    fn __moveinit__(out self, var other: Self):
        self.getter = other.getter^

    fn device(self) -> Int:
        return self.getter.device()

    fn stream(self) -> CUDAStreamType:
        return self.getter.stream()

    fn streamPtr(self) -> SharedStreamPtr:
        return self.getter.streamPtr()

    fn get[T: Movable & Typeable](mut self, ref data: Product[T]) raises -> ref [data] T:
        return self.getter.get[T](data)

    fn get[
        T: Movable & Typeable
    ](mut self, ref event: Event, ref token: EDGetTokenT[Product[T]]) raises -> ref [event] T:
        return self.getter.get[T](event, token)
