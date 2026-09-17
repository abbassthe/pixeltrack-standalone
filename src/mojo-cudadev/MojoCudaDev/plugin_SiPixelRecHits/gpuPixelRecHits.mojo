# Mojo port of plugin-SiPixelRecHits/gpuPixelRecHits.h. This is the GPU
# kernel, one CUDA block per active module, with ClusParams in shared memory.
from math import sqrt
from memory import stack_allocation
from std.memory import AddressSpace
from std.gpu.sync import barrier

import MojoCudaDev.CondFormats.PixelCPEforGPU as PixelCPEforGPU

from MojoCudaDev.CondFormats.PixelCPEforGPU import ParamsOnGPU, ClusParams
from MojoCudaDev.CUDACore.CUDAAtomics import (
    atomic_fetch_add,
    atomic_fetch_min,
    atomic_fetch_max,
)
from MojoCudaDev.CUDADataFormats.BeamSpotCUDA import BeamSpotPOD
from MojoCudaDev.CUDADataFormats.SiPixelClustersCUDA import (
    DeviceConstView as SiPixelClustersDeviceConstView,
)
from MojoCudaDev.CUDADataFormats.SiPixelDigisCUDA import (
    DeviceConstView as SiPixelDigisDeviceConstView,
)
from MojoCudaDev.CUDADataFormats.TrackingRecHit2DSOAView import (
    TrackingRecHit2DSOAView,
)
from MojoCudaDev.CUDADataFormats.gpuClusteringConstants import gpuClustering
from MojoCudaDev.DataFormats.ApproxAtan2 import ApproxAtan2
from MojoCudaDev.Geometry.Phase1PixelTopology import AverageGeometry
from MojoCudaDev.MojoBridge.DTypes import Float


struct gpuPixelRecHits:
    @staticmethod
    fn getHits(
        cpeParams: UnsafePointer[ParamsOnGPU, MutAnyOrigin],
        bs: UnsafePointer[BeamSpotPOD, MutAnyOrigin],
        pdigis: UnsafePointer[SiPixelDigisDeviceConstView, MutAnyOrigin],
        numElements: Int32,
        pclusters: UnsafePointer[SiPixelClustersDeviceConstView, MutAnyOrigin],
        phits: UnsafePointer[TrackingRecHit2DSOAView, MutAnyOrigin],
    ):
        from std.gpu import block_dim, block_idx, thread_idx

        # C++: __shared__ ClusParams clusParams.
        var clus_params_shared = stack_allocation[
            1, ClusParams, address_space=AddressSpace.SHARED
        ]()
        var clus_params = clus_params_shared.address_space_cast[
            AddressSpace.GENERIC
        ]()

        # C++ corrects this output geometry once, using the first module's
        # block. It is independent of the shared per-module accumulation.
        if block_idx.x == 0:
            var ag = cpeParams[].m_averageGeometry
            var agc = phits[].m_averageGeometry
            var il = Int(thread_idx.x)
            while il < Int(AverageGeometry.numberOfLaddersInBarrel):
                agc[].ladderZ[il] = ag[].ladderZ[il] - bs[].z
                agc[].ladderX[il] = ag[].ladderX[il] - bs[].x
                agc[].ladderY[il] = ag[].ladderY[il] - bs[].y
                agc[].ladderR[il] = sqrt(
                    agc[].ladderX[il] * agc[].ladderX[il]
                    + agc[].ladderY[il] * agc[].ladderY[il]
                )
                agc[].ladderMinZ[il] = ag[].ladderMinZ[il] - bs[].z
                agc[].ladderMaxZ[il] = ag[].ladderMaxZ[il] - bs[].z
                il += Int(block_dim.x)
            if thread_idx.x == 0:
                agc[].endCapZ[0] = ag[].endCapZ[0] - bs[].z
                agc[].endCapZ[1] = ag[].endCapZ[1] - bs[].z

        var module = Int(block_idx.x)
        var me = pclusters[].moduleId(module)
        var nclus = Int(pclusters[].clusInModule(Int(me)))
        if nclus == 0:
            return

        var start_clus = 0
        while start_clus < nclus:
            var nclus_in_iter = min(
                Int(PixelCPEforGPU.MaxHitsInIter), nclus - start_clus
            )
            var last_clus = start_clus + nclus_in_iter

            # Initialize the shared cluster accumulator.
            var ic = Int(thread_idx.x)
            while ic < nclus_in_iter:
                clus_params[].minRow[ic] = UInt32(0xFFFFFFFF)
                clus_params[].maxRow[ic] = 0
                clus_params[].minCol[ic] = UInt32(0xFFFFFFFF)
                clus_params[].maxCol[ic] = 0
                clus_params[].charge[ic] = 0
                clus_params[].q_f_X[ic] = 0
                clus_params[].q_l_X[ic] = 0
                clus_params[].q_f_Y[ic] = 0
                clus_params[].q_l_Y[ic] = 0
                ic += Int(block_dim.x)
            barrier()

            # One thread per digi: find the row/column extent of every
            # cluster in this MaxHitsInIter-sized batch.
            var first = Int(pclusters[].moduleStart(1 + module)) + Int(thread_idx.x)
            var i = first
            while i < Int(numElements):
                var id = pdigis[].moduleInd(i)
                if id != gpuClustering.invalidModuleId:
                    if id != UInt16(me):
                        break
                    var cluster = Int(pdigis[].clus(i))
                    if cluster >= start_clus and cluster < last_clus:
                        cluster -= start_clus
                        var x = UInt32(pdigis[].xx(i))
                        var y = UInt32(pdigis[].yy(i))
                        _ = atomic_fetch_min(
                            UnsafePointer(to=clus_params[].minRow[cluster]), x
                        )
                        _ = atomic_fetch_max(
                            UnsafePointer(to=clus_params[].maxRow[cluster]), x
                        )
                        _ = atomic_fetch_min(
                            UnsafePointer(to=clus_params[].minCol[cluster]), y
                        )
                        _ = atomic_fetch_max(
                            UnsafePointer(to=clus_params[].maxCol[cluster]), y
                        )
                i += Int(block_dim.x)
            barrier()

            # pixmx is unavailable in the binary dumps, matching C++'s
            # uint16_t maximum fallback.
            i = first
            while i < Int(numElements):
                var id = pdigis[].moduleInd(i)
                if id != gpuClustering.invalidModuleId:
                    if id != UInt16(me):
                        break
                    var cluster = Int(pdigis[].clus(i))
                    if cluster >= start_clus and cluster < last_clus:
                        cluster -= start_clus
                        var x = UInt32(pdigis[].xx(i))
                        var y = UInt32(pdigis[].yy(i))
                        var charge = Int32(pdigis[].adc(i))
                        _ = atomic_fetch_add(
                            UnsafePointer(to=clus_params[].charge[cluster]), charge
                        )
                        if clus_params[].minRow[cluster] == x:
                            _ = atomic_fetch_add(
                                UnsafePointer(to=clus_params[].q_f_X[cluster]), charge
                            )
                        if clus_params[].maxRow[cluster] == x:
                            _ = atomic_fetch_add(
                                UnsafePointer(to=clus_params[].q_l_X[cluster]), charge
                            )
                        if clus_params[].minCol[cluster] == y:
                            _ = atomic_fetch_add(
                                UnsafePointer(to=clus_params[].q_f_Y[cluster]), charge
                            )
                        if clus_params[].maxCol[cluster] == y:
                            _ = atomic_fetch_add(
                                UnsafePointer(to=clus_params[].q_l_Y[cluster]), charge
                            )
                i += Int(block_dim.x)
            barrier()

            # One thread per cluster: apply the CPE and write the rechit SOA.
            var first_hit = Int(pclusters[].clusModuleStart(Int(me))) + start_clus
            var common_params = cpeParams[].m_commonParams
            var det_params = cpeParams[].m_detParams + Int(me)
            ic = Int(thread_idx.x)
            while ic < nclus_in_iter:
                var hit = first_hit + ic
                debug_assert(hit < Int(phits[].m_nHits))
                debug_assert(
                    hit < Int(pclusters[].clusModuleStart(Int(me) + 1))
                )

                PixelCPEforGPU.position(
                    common_params[], det_params[], clus_params[], UInt32(ic)
                )
                PixelCPEforGPU.errorFromDB(
                    common_params[], det_params[], clus_params[], UInt32(ic)
                )

                phits[].m_charge[hit] = clus_params[].charge[ic]
                phits[].m_detInd[hit] = UInt16(me)

                var xl = clus_params[].xpos[ic]
                var yl = clus_params[].ypos[ic]
                phits[].m_xl[hit] = xl
                phits[].m_yl[hit] = yl
                phits[].m_xsize[hit] = clus_params[].xsize[ic]
                phits[].m_ysize[hit] = clus_params[].ysize[ic]
                phits[].m_xerr[hit] = clus_params[].xerr[ic] * clus_params[].xerr[ic]
                phits[].m_yerr[hit] = clus_params[].yerr[ic] * clus_params[].yerr[ic]

                var xg: Float = 0.0
                var yg: Float = 0.0
                var zg: Float = 0.0
                det_params[].frame.toGlobal(xl, yl, xg, yg, zg)
                xg -= bs[].x
                yg -= bs[].y
                zg -= bs[].z

                phits[].m_xg[hit] = xg
                phits[].m_yg[hit] = yg
                phits[].m_zg[hit] = zg
                phits[].m_rg[hit] = sqrt(xg * xg + yg * yg)
                phits[].m_iphi[hit] = ApproxAtan2.unsafe_atan2s[7](yg, xg)
                ic += Int(block_dim.x)
            barrier()

            start_clus += Int(PixelCPEforGPU.MaxHitsInIter)
