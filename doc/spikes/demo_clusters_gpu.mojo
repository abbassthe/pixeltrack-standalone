# Demo: real GPU execution through the ported SiPixelClustersCUDA data format.
#
# Allocates four device arrays + a device-resident DeviceConstView, fills the
# arrays from a CUDA kernel through the raw accessors, then reads them back
# from a second kernel through ONLY the view's own accessors -- which proves
# the host-staged view really points at the live device buffers.
#
# Run:  pixi run mojo run -I . -Xlinker -l:libcuda.so.1 ../../doc/spikes/demo_clusters_gpu.mojo
from MojoCudaDev.CUDADataFormats.SiPixelClustersCUDA import SiPixelClustersCUDA, DeviceConstView
from MojoCudaDev.CUDACore.allocate_device import _AllocateDeviceState
from MojoCudaDev.CUDACore.allocate_host import _AllocateHostState
from MojoCudaDev.CUDACore.host_unique_ptr import make_host_unique
from MojoCudaDev.CUDACore.CUDACompat import cudaStreamDefault
from std.gpu.host import DeviceContext


alias MAXMODULES = 8
alias TOTAL = (MAXMODULES + 1) * 2 + MAXMODULES * 2


fn check(name: String, ok: Bool):
    if ok:
        print("  PASS  ", name)
    else:
        print("  FAIL  ", name)


fn fill_kernel(
    moduleStart: UnsafePointer[UInt32, MutAnyOrigin],
    clusInModule: UnsafePointer[UInt32, MutAnyOrigin],
    moduleId: UnsafePointer[UInt32, MutAnyOrigin],
    clusModuleStart: UnsafePointer[UInt32, MutAnyOrigin],
):
    for i in range(MAXMODULES + 1):
        moduleStart[i] = UInt32(100 + i)
        clusModuleStart[i] = UInt32(400 + i)
    for i in range(MAXMODULES):
        clusInModule[i] = UInt32(200 + i)
        moduleId[i] = UInt32(300 + i)


fn read_via_view_kernel(
    view: UnsafePointer[DeviceConstView, MutAnyOrigin],
    out_ptr: UnsafePointer[UInt32, MutAnyOrigin],
):
    var o = 0
    for i in range(MAXMODULES + 1):
        out_ptr[o] = view[0].moduleStart(i)
        o += 1
    for i in range(MAXMODULES):
        out_ptr[o] = view[0].clusInModule(i)
        o += 1
    for i in range(MAXMODULES):
        out_ptr[o] = view[0].moduleId(i)
        o += 1
    for i in range(MAXMODULES + 1):
        out_ptr[o] = view[0].clusModuleStart(i)
        o += 1


def main():
    try:
        var ctx = DeviceContext(api="cuda")
        print("GPU device:", ctx.name())
        print("")

        var dev_state = _AllocateDeviceState()
        var host_state = _AllocateHostState()

        print("SiPixelClustersCUDA -- default construction")
        var empty = SiPixelClustersCUDA()
        check(
            "all pointers null, nClusters 0",
            Int(empty.moduleStart()) == 0
            and Int(empty.clusInModule()) == 0
            and Int(empty.moduleId()) == 0
            and Int(empty.clusModuleStart()) == 0
            and Int(empty.view()) == 0
            and empty.nClusters() == 0,
        )

        print("")
        print("SiPixelClustersCUDA -- device allocation (maxModules =", MAXMODULES, ")")
        var clusters = SiPixelClustersCUDA(
            UInt(MAXMODULES), dev_state, host_state, cudaStreamDefault
        )
        check(
            "four arrays + view allocated on device",
            Int(clusters.moduleStart()) != 0
            and Int(clusters.clusInModule()) != 0
            and Int(clusters.moduleId()) != 0
            and Int(clusters.clusModuleStart()) != 0
            and Int(clusters.view()) != 0,
        )
        check(
            "arrays are distinct allocations",
            Int(clusters.moduleStart()) != Int(clusters.clusInModule())
            and Int(clusters.clusInModule()) != Int(clusters.moduleId())
            and Int(clusters.moduleId()) != Int(clusters.clusModuleStart()),
        )

        print("")
        print("Launching fill kernel on the device...")
        var fill = ctx.compile_function[fill_kernel, fill_kernel]()
        ctx.enqueue_function(
            fill,
            clusters.moduleStart(),
            clusters.clusInModule(),
            clusters.moduleId(),
            clusters.clusModuleStart(),
            grid_dim=1,
            block_dim=1,
        )
        ctx.synchronize()

        print("Reading back through the device-resident view...")
        var out = make_host_unique[UInt32](UInt(TOTAL), host_state)
        var read = ctx.compile_function[read_via_view_kernel, read_via_view_kernel]()
        ctx.enqueue_function(
            read, clusters.view(), out[].get(), grid_dim=1, block_dim=1
        )
        ctx.synchronize()

        var p = out[].get()
        var ok = True
        var o = 0
        for i in range(MAXMODULES + 1):
            if p[o] != UInt32(100 + i):
                ok = False
            o += 1
        for i in range(MAXMODULES):
            if p[o] != UInt32(200 + i):
                ok = False
            o += 1
        for i in range(MAXMODULES):
            if p[o] != UInt32(300 + i):
                ok = False
            o += 1
        for i in range(MAXMODULES + 1):
            if p[o] != UInt32(400 + i):
                ok = False
            o += 1
        check(String(TOTAL) + " values exact via DeviceConstView accessors", ok)

        print("")
        print("SiPixelClustersCUDA -- accessors and move")
        clusters.setNClusters(1234)
        check("setNClusters / nClusters round-trip", clusters.nClusters() == 1234)

        var ms = Int(clusters.moduleStart())
        var vw = Int(clusters.view())
        var n = clusters.nClusters()
        var moved = clusters^
        check(
            "move preserves pointers and nClusters",
            Int(moved.moduleStart()) == ms
            and Int(moved.view()) == vw
            and moved.nClusters() == n,
        )
        print("")
        print("Done.")
    except e:
        print("EXCEPTION:", e)
