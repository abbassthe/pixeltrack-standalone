# Mojo port of plugin-SiPixelRecHits/PixelRecHitGPUKernel.{h,cu}.
from std.gpu.host import DeviceContext
from std.gpu.host.dim import Dim

from MojoCudaDev.CondFormats.PixelCPEforGPU import ParamsOnGPU
from MojoCudaDev.CUDACore.CUDACompat import CUDAStreamType, cudaStreamDefault
from MojoCudaDev.CUDACore.HistoContainer import fillManyFromVector
from MojoCudaDev.CUDACore.allocate_device import _AllocateDeviceState
from MojoCudaDev.CUDACore.allocate_host import _AllocateHostState
from MojoCudaDev.CUDACore.currentDevice import currentDevice
from MojoCudaDev.CUDADataFormats.BeamSpotCUDA import BeamSpotCUDA
from MojoCudaDev.CUDADataFormats.SiPixelClustersCUDA import SiPixelClustersCUDA
from MojoCudaDev.CUDADataFormats.SiPixelDigisCUDA import SiPixelDigisCUDA
from MojoCudaDev.CUDADataFormats.TrackingRecHit2DHeterogeneous import (
    TrackingRecHit2DHeterogeneous,
)
from MojoCudaDev.plugin_SiPixelRecHits.gpuPixelRecHits import gpuPixelRecHits
from MojoCudaDev.MojoBridge.DTypes import Typeable


fn _setHitsLayerStart(
    hits_module_start: UnsafePointer[UInt32, MutAnyOrigin],
    cpe_params: UnsafePointer[ParamsOnGPU, MutAnyOrigin],
    hits_layer_start: UnsafePointer[UInt32, MutAnyOrigin],
):
    from std.gpu import block_dim, block_idx, thread_idx

    var i = Int(block_dim.x) * Int(block_idx.x) + Int(thread_idx.x)
    if i < 11:
        var layer_geometry = cpe_params[].m_layerGeometry
        hits_layer_start[i] = hits_module_start[
            Int(layer_geometry[].layerStart[i])
        ]


struct PixelRecHitGPUKernel(Defaultable, Movable, Typeable):
    fn __init__(out self):
        pass

    fn makeHitsAsync(
        self,
        digis_d: SiPixelDigisCUDA,
        clusters_d: SiPixelClustersCUDA,
        bs_d: BeamSpotCUDA,
        cpe_params: UnsafePointer[ParamsOnGPU, MutAnyOrigin],
        mut dev_state: _AllocateDeviceState,
        mut host_state: _AllocateHostState,
        stream: CUDAStreamType = cudaStreamDefault,
    ) raises -> TrackingRecHit2DHeterogeneous:
        var n_hits = clusters_d.nClusters()
        var hits_d = TrackingRecHit2DHeterogeneous(
            n_hits,
            cpe_params,
            clusters_d.clusModuleStart(),
            dev_state,
            host_state,
            stream,
        )

        var ctx = DeviceContext(api="cuda", device_id=currentDevice())
        var device_stream = stream.get()

        # C++: one 128-thread block per active module, with getHits' shared
        # ClusParams accumulator handling one module in each block.
        if digis_d.nModules() != 0:
            var get_hits = ctx.compile_function[
                gpuPixelRecHits.getHits, gpuPixelRecHits.getHits
            ]()
            device_stream.enqueue_function(
                get_hits,
                cpe_params,
                bs_d.data(),
                digis_d.view(),
                Int32(digis_d.nDigis()),
                clusters_d.view(),
                hits_d.view(),
                grid_dim=Dim(Int(digis_d.nModules())),
                block_dim=Dim(128),
            )

        if n_hits != 0:
            var set_layer_start = ctx.compile_function[
                _setHitsLayerStart, _setHitsLayerStart
            ]()
            device_stream.enqueue_function(
                set_layer_start,
                clusters_d.clusModuleStart(),
                cpe_params,
                hits_d.hitsLayerStart(),
                grid_dim=Dim(1),
                block_dim=Dim(32),
            )

            # C++: fillManyFromVector(phiBinner, 10, iphi, hitsLayerStart,
            # nHits, 256, phiBinnerStorage, stream).
            fillManyFromVector[DType.int16, 128, -1, 16, 10](
                hits_d.phiBinner(),
                10,
                hits_d.iphi(),
                hits_d.hitsLayerStart(),
                Int32(n_hits),
                256,
                hits_d.phiBinnerStorage(),
                stream,
            )

        return hits_d^

    @staticmethod
    fn dtype() -> String:
        return "PixelRecHitGPUKernel"
