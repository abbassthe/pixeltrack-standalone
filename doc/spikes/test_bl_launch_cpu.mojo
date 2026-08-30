# Drives HelixFitOnGPU::launchBrokenLineKernelsOnCPU (the port of
# BrokenLineFitOnGPU.cc) end-to-end over one hand-built 4-hit tuple on a known
# circle, checking the fitted pt/eta land in the output SoA.
#
# Run with -D GPU_SMALL_EVENTS so TrackSoA is 2K rather than 32K entries.
import math

from MojoCudaDev.CondFormats.PixelCPEforGPU import (
    ParamsOnGPU,
    CommonParams,
    DetParams,
    LayerGeometry,
)
from MojoCudaDev.CUDACore.CUDACompat import cudaStreamDefault
from MojoCudaDev.CUDADataFormats.TrackingRecHit2DSOAView import (
    TrackingRecHit2DSOAView,
)
from MojoCudaDev.CUDADataFormats.TrackSoAHeterogeneousT import pixelTrack
from MojoCudaDev.Geometry.Phase1PixelTopology import AverageGeometry
from MojoCudaDev.plugin_PixelTriplets.CAConstants import caConstants
from MojoCudaDev.plugin_PixelTriplets.HelixFitOnGpu import HelixFitOnGPU
from MojoCudaDev.MojoBridge.DTypes import Float

comptime NHITS = 4
comptime CX = 3.0
comptime CY = 4.0
comptime RAD = 10.0
comptime BFIELD = 0.0114256972711507


fn main() raises:
    # ---- hits: NHITS points exactly on the circle, z linear in arc length ----
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

    var common = CommonParams()
    var det = List[DetParams](length=1, fill=DetParams())
    # a default SOARotation is all-zero, not identity: leaving it would send
    # zero errors into lineFit/circleFit and every fitted value comes back NaN
    det[0].frame.rot.R11 = 1
    det[0].frame.rot.R22 = 1
    det[0].frame.rot.R33 = 1
    var layerGeom = LayerGeometry()
    var avgGeom = AverageGeometry()

    var cpe = ParamsOnGPU()
    cpe.m_commonParams = UnsafePointer(to=common)
    cpe.m_detParams = det.unsafe_ptr()
    cpe.m_layerGeometry = UnsafePointer(to=layerGeom)
    cpe.m_averageGeometry = UnsafePointer(to=avgGeom)

    var hh = TrackingRecHit2DSOAView()
    hh.m_xg = xg.unsafe_ptr()
    hh.m_yg = yg.unsafe_ptr()
    hh.m_zg = zg.unsafe_ptr()
    hh.m_xerr = xerr.unsafe_ptr()
    hh.m_yerr = yerr.unsafe_ptr()
    hh.m_detInd = detInd.unsafe_ptr()
    hh.m_cpeParams = UnsafePointer(to=cpe)
    hh.m_nHits = NHITS

    # ---- one tuple (tkid 0) of NHITS hits ----
    var tuples = pixelTrack.HitContainer()
    tuples.off[0] = 0
    for b in range(1, tuples.totOnes()):
        tuples.off[b] = NHITS
    for i in range(NHITS):
        tuples.content[i] = UInt32(i)

    var tm = caConstants.TupleMultiplicity()
    for b in range(tm.totOnes()):
        tm.off[b] = 0 if b <= NHITS else 1
    tm.content[0] = caConstants.tindex_type(0)

    # ---- output SoA on the heap: too large for the stack ----
    var out = alloc[pixelTrack.TrackSoA](1)
    __get_address_as_uninit_lvalue(out.address) = pixelTrack.TrackSoA()

    # ---- the real thing: BrokenLineFitOnGPU.cc's launcher ----
    var fitter = HelixFitOnGPU(Float32(BFIELD), False)
    fitter.allocateOnGPU(
        UnsafePointer(to=tuples), UnsafePointer(to=tm), out
    )
    fitter.launchBrokenLineKernelsOnCPU(
        UnsafePointer(to=hh), UInt32(NHITS), UInt32(1)
    )

    var pt = out[].pt[0]
    var eta = out[].eta[0]
    var chi2 = out[].chi2[0]
    print("track 0:  pt =", pt, " eta =", eta, " chi2 =", chi2)

    # pt = bField / |k| and the hits sit on radius RAD, so |k| = 1/RAD
    var expected_pt = Float32(BFIELD * RAD)
    # cot(theta) = 2 by construction -> eta = asinh(2)
    var expected_eta = Float32(math.asinh(2.0))
    print("expected: pt =", expected_pt, " eta =", expected_eta)

    var ok = (
        abs(pt - expected_pt) < 1e-3 * expected_pt
        and abs(eta - expected_eta) < 1e-3
    )
    if ok:
        print("LAUNCH_CPU OK")
    else:
        print("LAUNCH_CPU MISMATCH")

    out.destroy_pointee()
