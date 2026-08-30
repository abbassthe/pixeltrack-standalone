# Drives kernelBLFastFit on the host path (block_dim/grid_dim collapse to 1,
# so first=0 stride=1) over one hand-built 4-hit tuple lying exactly on a
# known circle, and checks the fast-fit recovers that circle.
import math

from MojoCudaDev.CondFormats.PixelCPEforGPU import (
    ParamsOnGPU,
    CommonParams,
    DetParams,
    LayerGeometry,
)
from MojoCudaDev.CUDADataFormats.TrackingRecHit2DSOAView import (
    TrackingRecHit2DSOAView,
)
from MojoCudaDev.CUDADataFormats.TrackSoAHeterogeneousT import pixelTrack
from MojoCudaDev.Geometry.Phase1PixelTopology import AverageGeometry
from MojoCudaDev.plugin_PixelTriplets.BrokenLineFitOnGPU import kernelBLFastFit
from MojoCudaDev.plugin_PixelTriplets.CAConstants import caConstants
from MojoCudaDev.plugin_PixelTriplets.HelixFitOnGpu import riemannFit
from MojoCudaDev.MojoBridge.DTypes import Float

comptime NHITS = 4
comptime F = DType.float64

# the circle the hits are generated on
comptime CX = 3.0
comptime CY = 4.0
comptime RAD = 10.0


fn main() raises:
    comptime nConcurrent = Int(riemannFit.maxNumberOfConcurrentFits)

    # ---- hit arrays: NHITS points exactly on the circle ----
    var xg = List[Float](length=NHITS, fill=0)
    var yg = List[Float](length=NHITS, fill=0)
    var zg = List[Float](length=NHITS, fill=0)
    var xerr = List[Float](length=NHITS, fill=0)
    var yerr = List[Float](length=NHITS, fill=0)
    var detInd = List[UInt16](length=NHITS, fill=0)

    for i in range(NHITS):
        var phi = 0.20 + 0.13 * Float64(i)
        xg[i] = Float(CX + RAD * math.cos(phi))
        yg[i] = Float(CY + RAD * math.sin(phi))
        zg[i] = Float(2.0 * phi * RAD)
        xerr[i] = Float(2.5e-5)
        yerr[i] = Float(2.5e-5)
        detInd[i] = 0

    # ---- CPE params: one DetParams, default (identity) frame ----
    var common = CommonParams()
    var det = List[DetParams](length=1, fill=DetParams())
    var layerGeom = LayerGeometry()
    var avgGeom = AverageGeometry()

    var cpe = ParamsOnGPU()
    cpe.m_commonParams = UnsafePointer(to=common)
    cpe.m_detParams = det.unsafe_ptr()
    cpe.m_layerGeometry = UnsafePointer(to=layerGeom)
    cpe.m_averageGeometry = UnsafePointer(to=avgGeom)

    # ---- the hit view ----
    var hh = TrackingRecHit2DSOAView()
    hh.m_xg = xg.unsafe_ptr()
    hh.m_yg = yg.unsafe_ptr()
    hh.m_zg = zg.unsafe_ptr()
    hh.m_xerr = xerr.unsafe_ptr()
    hh.m_yerr = yerr.unsafe_ptr()
    hh.m_detInd = detInd.unsafe_ptr()
    hh.m_cpeParams = UnsafePointer(to=cpe)
    hh.m_nHits = NHITS

    # ---- foundNtuplets: exactly one tuple (tkid 0) holding hits 0..NHITS-1 ----
    var tuples = pixelTrack.HitContainer()
    tuples.off[0] = 0
    for b in range(1, tuples.totOnes()):
        tuples.off[b] = NHITS
    for i in range(NHITS):
        tuples.content[i] = UInt32(i)

    # ---- tupleMultiplicity: bin NHITS holds one entry, pointing at tkid 0 ----
    var tm = caConstants.TupleMultiplicity()
    for b in range(tm.totOnes()):
        tm.off[b] = 0 if b <= NHITS else 1
    tm.content[0] = caConstants.tindex_type(0)

    print("tuples.nOnes      =", tuples.nOnes())
    print("tuples.size(0)    =", tuples.size(0))
    print("tm.size(NHITS)    =", tm.size(NHITS))

    # ---- kernel scratch, sized as the C++ launcher sizes it ----
    var phits = List[Float64](length=nConcurrent * 3 * 4, fill=0.0)
    var phits_ge = List[Float32](length=nConcurrent * 6 * 4, fill=0.0)
    var pfast_fit = List[Float64](length=nConcurrent * 4, fill=0.0)

    kernelBLFastFit[NHITS](
        UnsafePointer(to=tuples),
        UnsafePointer(to=tm),
        UnsafePointer(to=hh),
        phits.unsafe_ptr(),
        phits_ge.unsafe_ptr(),
        pfast_fit.unsafe_ptr(),
        UInt32(NHITS),
        UInt32(0),
    )

    # fast_fit is a Map4d over pfast_fit with inner stride = maxNumberOfConcurrentFits
    var stride = nConcurrent
    var fx = pfast_fit[0 * stride]
    var fy = pfast_fit[1 * stride]
    var fr = pfast_fit[2 * stride]
    var fc = pfast_fit[3 * stride]
    print("")
    print("fast_fit centre =", fx, fy, " radius =", fr, " cot(theta) =", fc)
    print("expected centre =", CX, CY, " radius =", RAD)

    # hits reach the kernel through the SOAView as Float32 (C++ does the same),
    # so ~1e-7 relative on the inputs is the floor here.
    comptime TOL = 1e-3
    var ok = (
        abs(fx - CX) < TOL and abs(fy - CY) < TOL and abs(fr - RAD) < TOL
    )
    if ok:
        print("KERNEL OK")
    else:
        print("KERNEL MISMATCH")