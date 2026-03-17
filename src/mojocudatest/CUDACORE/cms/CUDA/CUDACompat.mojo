from gpu.host import DeviceContext
from gpu.host.device_context import DeviceStream

from MojoBridge.DTypes import (
    cudaError_t,
    cudaSuccess,
    cudaErrorNotReady,
)
from utils.lock import BlockingSpinLock, BlockingScopedLock


alias CUDAStreamType = DeviceStream
# Sentinel "default stream" value. We resolve this to the current default stream
# inside `cudaEventRecord()` before enqueueing the completion callback.
alias cudaStreamDefault: CUDAStreamType = CUDAStreamType()


@fieldwise_init
struct CUDAEventType(Copyable, Defaultable, Movable):
    # Copyable handle to event state stored in the module registry.
    # `event_id == 0` means "never recorded / no registry slot allocated yet".
    var event_id: UInt64

    @always_inline
    fn __init__(out self):
        self.event_id = 0


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
fn _default_device_stream() -> CUDAStreamType:
    var ctx = DeviceContext()
    return ctx.stream


@always_inline
fn _resolve_device_stream(stream: CUDAStreamType) -> CUDAStreamType:
    if stream == cudaStreamDefault:
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
    if event.event_id == 0:
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


fn _mark_event_completed_by_id(event_id: UInt64, seq: UInt64):
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


fn _complete_recorded_event(event_id: UInt64, seq: UInt64):
    _mark_event_completed_by_id(event_id, seq)


fn cudaGetDevice(out device: Int) -> cudaError_t:
    var ctx = DeviceContext()
    device = ctx.id()
    return cudaSuccess


fn cudaEventCreateWithFlags(out event: CUDAEventType, flags: UInt32) -> cudaError_t:
    _ = flags
    event = CUDAEventType()
    _ensure_event_slot(event)
    if event.event_id == 0:
        return cudaErrorNotReady
    return cudaSuccess


fn cudaEventDestroy(mut event: CUDAEventType) -> cudaError_t:
    if event.event_id == 0:
        return cudaSuccess

    with BlockingScopedLock(_cuda_event_registry_lock):
        let idx = _event_index_from_id(event.event_id)
        if idx >= 0 and idx < _cuda_event_registry.__len__():
            _cuda_event_registry[idx] = _CUDAEventState()
    event.event_id = 0
    return cudaSuccess


fn cudaEventRecord(mut event: CUDAEventType, stream: CUDAStreamType) -> cudaError_t:
    let record_seq = _record_event_seq(event)
    if event.event_id == 0 or record_seq == 0:
        return cudaErrorNotReady

    var actual_stream = _resolve_device_stream(stream)
    # Run host-side completion update in stream order, after all previously queued work.
    actual_stream.enqueue_function[
        _complete_recorded_event,
        event_id = event.event_id,
        seq = record_seq
    ]()
    return cudaSuccess


fn cudaEventQuery(event: read CUDAEventType) -> cudaError_t:
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
