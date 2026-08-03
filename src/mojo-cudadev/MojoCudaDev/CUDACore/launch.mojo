# Mojo port of CUDACore/launch.h.
#
# `LaunchParameters` bundles a launch config (grid, block, shared memory,
# stream) exactly like the C++ struct. There is no Mojo equivalent of
# `cms::cuda::launch(kernel, config, args...)` as a callable wrapper -- tried
# two different designs, both fail for independent reasons:
#
#   1. `DeviceContext.compile_function[func, func]()` (needed to resolve a bare
#      kernel reference) requires `func` named literally at that exact call
#      site to introspect its argument types; forwarding it through a generic
#      parameter errors ("'signature_func' ... but value has type 'func_type'")
#      because the concrete parameter-type list has already been erased.
#   2. Even routing around #1 by accepting an already-compiled `DeviceFunction`
#      (so the wrapper only has to forward the *argument* pack, not the
#      kernel) hits a second, independent wall: forwarding a variadic pack
#      into a nested call is not yet implemented --
#      `stream.enqueue_function(compiled, *args, ...)` inside a wrapper
#      function errors with "unpacked arguments are not supported yet".
#
# Neither is a design mistake to fix later -- #2 is an explicit "not supported
# yet" from the compiler itself. This costs nothing in practice: every
# cudadev call site already names its kernel literally, never through
# indirection. The call-site idiom is:
#
#   var compiled = ctx.compile_function[kernel, kernel]()
#   config.stream.get().enqueue_function(
#       compiled, arg0, arg1, ...,
#       grid_dim=config.grid_dim, block_dim=config.block_dim,
#       shared_mem_bytes=config.shared_mem,
#   )
#
# `launch_cooperative` (cudaLaunchCooperativeKernel) is not ported: `cudadev`
# never calls it, only defines it, and the same two constraints above apply to
# it identically (confirmed). Cooperative launch itself is just an
# `enqueue_function` call carrying an extra attribute if it's ever needed:
#   from std.gpu.host.launch_attribute import (
#       LaunchAttribute, LaunchAttributeID, LaunchAttributeValue,
#   )
#   ... attributes=[LaunchAttribute(
#       LaunchAttributeID.COOPERATIVE, LaunchAttributeValue(True)
#   )]
from std.gpu.host import DeviceContext
from std.gpu.host.dim import Dim

from MojoCudaDev.CUDACore.CUDACompat import CUDAStreamType, cudaStreamDefault


struct LaunchParameters(Copyable, Movable):
    var grid_dim: Dim
    var block_dim: Dim
    var shared_mem: Int
    var stream: CUDAStreamType

    @always_inline
    fn __init__(
        out self,
        grid_dim: Dim,
        block_dim: Dim,
        shared_mem: Int = 0,
        stream: CUDAStreamType = cudaStreamDefault,
    ):
        self.grid_dim = grid_dim
        self.block_dim = block_dim
        self.shared_mem = shared_mem
        self.stream = stream
