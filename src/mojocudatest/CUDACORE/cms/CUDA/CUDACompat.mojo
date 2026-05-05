from std.gpu.host import DeviceContext
from std.gpu.host.device_context import DeviceEvent, DeviceStream


from MojoBridge.DTypes import (
    cudaError_t,
    cudaSuccess,
    cudaErrorNotReady,
)

from utils.lock import BlockingSpinLock, BlockingScopedLock


var _cuda_stream_id_lock = BlockingSpinLock()
var _cuda_next_stream_id: UInt64 = 1


fn _next_cuda_stream_id() -> UInt64:
    with BlockingScopedLock(_cuda_stream_id_lock):
        let stream_id = _cuda_next_stream_id
        _cuda_next_stream_id += 1
        return stream_id


@fieldwise_init
struct CUDAStreamType(ImplicitlyCopyable, Defaultable, Movable):
    var stream_id: UInt64
    var stream: Optional[DeviceStream]

    @always_inline
    fn __init__(out self):
        self.stream_id = 0
        self.stream = None

    @always_inline
    fn __init__(out self, stream: DeviceStream):
        self.stream = stream

    @always_inline
    fn is_default(self) -> Bool:
        return self.stream_id == 0

    @always_inline
    fn get(self) raises -> DeviceStream:
        if self.stream:
            return self.stream.value()
        var ctx = DeviceContext()
        return ctx.stream()

    @always_inline
    fn record_event(self, event: DeviceEvent) raises:
        var actual_stream = self.get()
        actual_stream.record_event(event)

    @always_inline
    fn enqueue_wait_for(self, event: DeviceEvent) raises:
        var actual_stream = self.get()
        actual_stream.enqueue_wait_for(event)

    @always_inline
    fn synchronize(self) raises:
        var actual_stream = self.get()
        actual_stream.synchronize()

    @always_inline
    fn __eq__(self, other: Self) -> Bool:
        return self.stream_id == other.stream_id

    @always_inline
    fn __ne__(self, other: Self) -> Bool:
        return self.stream_id != other.stream_id


# Sentinel "default stream" value. We resolve this to the current default stream
# inside CUDA compatibility calls.
comptime cudaStreamDefault: CUDAStreamType = CUDAStreamType()


@fieldwise_init
struct CUDAEventType(Copyable, Defaultable, Movable):
    # Copyable handle to event state stored in the module registry.
    # `event_id == 0` means "never recorded / no registry slot allocated yet".
    var event_id: UInt64

    # Real stream-side state for Mojo GPU synchronization. event_id remains the
    # fake query handle used by cudaEventQuery().
    var has_real_event: Bool
    var real_event: Optional[DeviceEvent]

    @always_inline
    fn __init__(out self):
        self.event_id = 0
        self.has_real_event = False
        self.real_event = None


@fieldwise_init
struct _CUDAEventState(Copyable, Defaultable, Movable):
    var expected_seq: UInt64
    var completed_seq: UInt64

    @always_inline
    fn __init__(out self):
        self.expected_seq = 0
        self.completed_seq = 0


var _cuda_event_registry_lock = BlockingSpinLock()
var _cuda_event_registry = List[_CUDAEventState]()


@always_inline
fn _default_device_stream() raises -> CUDAStreamType:
    var ctx = DeviceContext()
    return CUDAStreamType(ctx.stream())


@always_inline
fn _resolve_device_stream(stream: CUDAStreamType) raises -> CUDAStreamType:
    if stream.is_default():
        return _default_device_stream()
    return stream


@always_inline
fn _event_index_from_id(event_id: UInt64) -> Int:
    return Int(event_id) - 1



fn _ensure_event_slot(mut event: CUDAEventType):
    if event.event_id != 0:
        return
    with BlockingScopedLock(_cuda_event_registry_lock):
        if event.event_id == 0:
            _cuda_event_registry.append(_CUDAEventState())
            event.event_id = UInt64(_cuda_event_registry.__len__())


fn _record_event_seq(mut event: CUDAEventType) -> UInt64:
    _ensure_event_slot(event)
    if event.event_id == 0 :
        return 0

    with BlockingScopedLock(_cuda_event_registry_lock):
        let idx = _event_index_from_id(event.event_id)
        if idx < 0 or idx >= _cuda_event_registry.__len__():
            return 0
        var state = _cuda_event_registry[idx]
        state.expected_seq += 1
        let seq = state.expected_seq
        _cuda_event_registry[idx] = state
        return seq


fn _mark_event_completed_by_id(mut event_id: UInt64, seq: UInt64):
    if event_id == 0:
        return
    with BlockingScopedLock(_cuda_event_registry_lock):
        let idx = _event_index_from_id(event_id)
        if idx < 0 or idx >= _cuda_event_registry.__len__():
            return
        var state = _cuda_event_registry[idx]
        # Keep monotonic progress in case stale callbacks complete after newer ones.
        if state.completed_seq < seq:
            state.completed_seq = seq
            _cuda_event_registry[idx] = state


fn _complete_recorded_event(mut event_id: UInt64, seq: UInt64):
    _mark_event_completed_by_id(event_id, seq)


fn _wait_for_recorded_event(event_id: UInt64):
    if event_id == 0 :
        return

    while True:
        with BlockingScopedLock(_cuda_event_registry_lock):
            let idx = _event_index_from_id(event_id)
            if idx < 0 or idx >= _cuda_event_registry.__len__():
                return
            let state = _cuda_event_registry[idx]
            if state.completed_seq == state.expected_seq:
                return


fn _ensure_real_event(mut event: CUDAEventType) -> Bool:
    if event.has_real_event and event.real_event:
        return True

    try:
        var ctx = DeviceContext()
        event.real_event = ctx.create_event()
        event.has_real_event = True
        return True
    except e:
        event.has_real_event = False
        event.real_event = None
        return False


fn cudaGetDevice(mut device: Int) -> cudaError_t:
    try:
        var ctx = DeviceContext()
        device = Int(ctx.id())
        return cudaSuccess
    except e:
        return cudaErrorNotReady


fn cudaSetDevice(device: Int) -> cudaError_t:
    try:
        _ = DeviceContext(device_id=device)
        return cudaSuccess
    except e:
        return cudaErrorNotReady


fn cudaEventCreateWithFlags(mut event: CUDAEventType, flags: UInt32) -> cudaError_t:
    _ = flags
    event = CUDAEventType()
    _ensure_event_slot(event)
    if event.event_id == 0:
        return cudaErrorNotReady
    _ = _ensure_real_event(event)
    return cudaSuccess


fn cudaEventDestroy(mut event: CUDAEventType) -> cudaError_t:
    event.has_real_event = False
    event.real_event = None

    if event.event_id == 0:
        return cudaSuccess

    with BlockingScopedLock(_cuda_event_registry_lock):
        let idx = _event_index_from_id(event.event_id)
        if idx >= 0 and idx < _cuda_event_registry.__len__():
            _cuda_event_registry[idx] = _CUDAEventState()
    event.event_id = 0
    return cudaSuccess


fn cudaEventRecord(mut event: CUDAEventType, stream: CUDAStreamType) -> cudaError_t:
    var record_seq = _record_event_seq(event)
    if event.event_id == 0 or record_seq == 0:
        return cudaErrorNotReady

    try:
        var actual_stream = _resolve_device_stream(stream)
        if _ensure_real_event(event):
            actual_stream.record_event(event.real_event.value())
    except e:
        return cudaErrorNotReady

    # Keep cudaEventQuery() nonblocking. Stable std.gpu.host exposes real
    # stream wait/record APIs, but not a nonblocking event query or host
    # completion callback equivalent to the old enqueue_function path.
    _mark_event_completed_by_id(event.event_id, record_seq)
    return cudaSuccess


fn cudaStreamWaitEvent(
    stream: CUDAStreamType, event: CUDAEventType, flags: UInt32
) -> cudaError_t:
    _ = flags
    if event.event_id == 0:
        return cudaSuccess

    var actual_stream: CUDAStreamType
    try:
        actual_stream = _resolve_device_stream(stream)
    except e:
        return cudaErrorNotReady
        
    if event.has_real_event and event.real_event:
        try:
            actual_stream.enqueue_wait_for(event.real_event.value())
            return cudaSuccess
        except e:
            return cudaErrorNotReady

    _wait_for_recorded_event(event.event_id)
    return cudaSuccess


fn cudaEventQuery(read event: CUDAEventType) -> cudaError_t:
    if event.event_id == 0:
        # Never recorded => ready.
        return cudaSuccess

    with BlockingScopedLock(_cuda_event_registry_lock):
        let idx = _event_index_from_id(event.event_id)
        if idx < 0 or idx >= _cuda_event_registry.__len__():
            return cudaErrorNotReady
        let state = _cuda_event_registry[idx]
        if state.completed_seq == state.expected_seq:
            return cudaSuccess
    return cudaErrorNotReady


fn cudaEventMarkCompleted(mut event: CUDAEventType) -> cudaError_t:
    # Test/helper hook for code paths that want to complete the latest record explicitly.
    if event.event_id == 0:
        return cudaSuccess

    with BlockingScopedLock(_cuda_event_registry_lock):
        let idx = _event_index_from_id(event.event_id)
        if idx < 0 or idx >= _cuda_event_registry.__len__():
            return cudaErrorNotReady
        var state = _cuda_event_registry[idx]
        state.completed_seq = state.expected_seq
        _cuda_event_registry[idx] = state
    return cudaSuccess
