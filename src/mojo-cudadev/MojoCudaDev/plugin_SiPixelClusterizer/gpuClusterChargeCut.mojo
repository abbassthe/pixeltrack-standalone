# Mojo port of plugin-SiPixelClusterizer/gpuClusterChargeCut.h.
from memory import stack_allocation
from std.memory import AddressSpace
from std.gpu.sync import barrier

from MojoCudaDev.CUDACore.CUDAAtomics import atomic_fetch_add
from MojoCudaDev.CUDACore.PrefixScan import blockPrefixScan
from MojoCudaDev.CUDADataFormats.gpuClusteringConstants import gpuClustering
from MojoCudaDev.Geometry.Phase1PixelTopology import Phase1PixelTopology
from MojoCudaDev.plugin_SiPixelClusterizer.SiPixelClusterThresholds import (
    SiPixelClusterThresholds,
)


struct gpuClusterChargeCut:
    # C++: gpuClustering::clusterChargeCut(). One block processes one active
    # module at a time, and grid-strides across active modules.
    @staticmethod
    fn clusterChargeCut(
        clusterThresholds: SiPixelClusterThresholds,
        id: UnsafePointer[UInt16, MutAnyOrigin],
        adc: UnsafePointer[UInt16, MutAnyOrigin],
        moduleStart: UnsafePointer[UInt32, MutAnyOrigin],
        nClustersInModule: UnsafePointer[UInt32, MutAnyOrigin],
        moduleId: UnsafePointer[UInt32, MutAnyOrigin],
        clusterId: UnsafePointer[Int32, MutAnyOrigin],
        numElements: UInt32,
    ):
        from std.gpu import block_dim, block_idx, grid_dim, thread_idx

        comptime maxClusters = Int(gpuClustering.maxNumClustersPerModules)
        var charge_sh = stack_allocation[
            maxClusters, Int32, address_space=AddressSpace.SHARED
        ]()
        var ok_sh = stack_allocation[
            maxClusters, UInt8, address_space=AddressSpace.SHARED
        ]()
        # C++ uses UInt16 here. PrefixScan's GPU shuffle implementation in
        # this Mojo release supports UInt32 but not UInt16, so only this
        # private scratch table is widened; cluster IDs remain Int32/UInt16
        # at their C++-matching interfaces.
        var newclus_id_sh = stack_allocation[
            maxClusters, UInt32, address_space=AddressSpace.SHARED
        ]()
        var ws = stack_allocation[
            32, UInt32, address_space=AddressSpace.SHARED
        ]()

        # PrefixScan accepts generic-space input pointers; the backing arrays
        # remain shared allocations exactly as in C++.
        var charge = charge_sh.address_space_cast[AddressSpace.GENERIC]()
        var ok = ok_sh.address_space_cast[AddressSpace.GENERIC]()
        var newclus_id = newclus_id_sh.address_space_cast[AddressSpace.GENERIC]()

        var module = Int(block_idx.x)
        var endModule = Int(moduleStart[0])
        while module < endModule:
            var firstPixel = Int(moduleStart[1 + module])
            var thisModuleId = id[firstPixel]
            debug_assert(
                Int(thisModuleId) < Int(gpuClustering.maxNumModules),
                "clusterChargeCut: module id out of range",
            )
            debug_assert(
                UInt32(thisModuleId) == moduleId[module],
                "clusterChargeCut: module-id table mismatch",
            )

            var nclus = nClustersInModule[Int(thisModuleId)]
            if nclus != 0:
                var first = firstPixel + Int(thread_idx.x)
                if nclus > UInt32(gpuClustering.maxNumClustersPerModules):
                    # Keep C++'s overflow policy: discard clusters above the
                    # bounded shared-memory capacity before applying the cut.
                    var i = first
                    while i < Int(numElements):
                        if id[i] != gpuClustering.invalidModuleId:
                            if id[i] != thisModuleId:
                                break
                            if clusterId[i] >= gpuClustering.maxNumClustersPerModules:
                                id[i] = gpuClustering.invalidModuleId
                                clusterId[i] = Int32(gpuClustering.invalidModuleId)
                        i += Int(block_dim.x)
                    nclus = UInt32(gpuClustering.maxNumClustersPerModules)

                debug_assert(
                    nclus <= UInt32(gpuClustering.maxNumClustersPerModules),
                    "clusterChargeCut: too many clusters",
                )
                var j = Int(thread_idx.x)
                while j < Int(nclus):
                    charge[j] = 0
                    j += Int(block_dim.x)
                barrier()

                var i = first
                while i < Int(numElements):
                    if id[i] != gpuClustering.invalidModuleId:
                        if id[i] != thisModuleId:
                            break
                        _ = atomic_fetch_add(charge + Int(clusterId[i]), Int32(adc[i]))
                    i += Int(block_dim.x)
                barrier()

                var chargeCut = clusterThresholds.getThresholdForLayerOnCondition(
                    thisModuleId < UInt16(Phase1PixelTopology.layerStart[1])
                )
                j = Int(thread_idx.x)
                while j < Int(nclus):
                    var keep = charge[j] > chargeCut
                    newclus_id[j] = UInt32(1) if keep else UInt32(0)
                    ok[j] = UInt8(1) if keep else UInt8(0)
                    j += Int(block_dim.x)
                barrier()

                blockPrefixScan(newclus_id, nclus, ws)

                var kept = newclus_id[Int(nclus) - 1]
                debug_assert(
                    nclus >= kept,
                    "clusterChargeCut: prefix scan exceeds input clusters",
                )
                if nclus != kept:
                    nClustersInModule[Int(thisModuleId)] = kept
                    barrier()

                    # Mark failed clusters as invalidModuleId + 1 so that the
                    # final `-1` produces exactly invalidModuleId, as C++.
                    j = Int(thread_idx.x)
                    while j < Int(nclus):
                        if ok[j] == 0:
                            newclus_id[j] = UInt32(gpuClustering.invalidModuleId) + 1
                        j += Int(block_dim.x)
                    barrier()

                    i = first
                    while i < Int(numElements):
                        if id[i] != gpuClustering.invalidModuleId:
                            if id[i] != thisModuleId:
                                break
                            clusterId[i] = Int32(newclus_id[Int(clusterId[i])]) - 1
                            if clusterId[i] == Int32(gpuClustering.invalidModuleId):
                                id[i] = gpuClustering.invalidModuleId
                        i += Int(block_dim.x)

            module += Int(grid_dim.x)
