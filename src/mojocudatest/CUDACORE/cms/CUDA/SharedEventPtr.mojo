from memory import Arc

from CUDACompat import CUDAEventType, cudaEventDestroy


# Owns one CUDA event and destroys it when the last Arc reference drops.
# Mirrors the role of the custom deleter in C++'s shared_ptr<cudaEvent_t>.
struct _EventOwner(Movable):
    var event: CUDAEventType

    fn __init__(out self, owned event: CUDAEventType):
        self.event = event^

    fn __moveinit__(out self, var other: Self):
        self.event = other.event

    fn __del__(var self):
        _ = cudaEventDestroy(self.event)


# Equivalent to cms::cuda::SharedEventPtr.
# In C++: shared_ptr<remove_pointer_t<cudaEvent_t>>
# In Mojo: Arc handles the reference count; _EventOwner handles destruction.
alias SharedEventPtr = Arc[_EventOwner]
