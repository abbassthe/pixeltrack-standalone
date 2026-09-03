from std.sys import size_of
import std.random as random
from std.utils.numerics import min_finite, max_finite

from MojoSerial.CUDACore.HistoContainer import HistoContainer
import MojoSerial.CUDACore.HistoContainer as Histo


def go[T: DType, NBINS: Int = 128, S: Int = 8 * size_of[T](), DELTA: Int = 1000]():
    random.seed()

    var rmin = min_finite[T]()
    var rmax = max_finite[T]()

    var rand_int = random.random_si64(Int64(rmin), Int64(rmax))

    if NBINS != 128:
        rmin = 0
        rmax = NBINS * 2 - 1

    comptime N: Int = 12000
    var v = InlineArray[Scalar[T], N](uninitialized=True)

    comptime Hist = HistoContainer[T, NBINS, N, S]
    comptime Hist4 = HistoContainer[T, NBINS, N, S, DType.uint16, 4]

    print(
        "HistoContainer ",
        Hist.nbits(),
        " ",
        Hist.nbins(),
        " ",
        Hist.totbins(),
        " ",
        Hist.capacity(),
        " ",
        Int(rmax - rmin) // Hist.nbins(),
        sep="",
    )
    print("bins ", Hist.bin(0), " ", Hist.bin(rmin), " ", Hist.bin(rmax))
    print(
        "HistoContainer4 ",
        Hist4.nbits(),
        " ",
        Hist4.nbins(),
        " ",
        Hist4.totbins(),
        " ",
        Hist4.capacity(),
        " ",
        Int(rmax - rmin) // Hist.nbins(),
        sep="",
    )

    for nh in range(4):
        print(
            "bins ",
            Int(Hist4.bin(0)) + Hist4.histOff(nh),
            " ",
            Int(Hist.bin(rmin)) + Hist4.histOff(nh),
            " ",
            Int(Hist.bin(rmax)) + Hist4.histOff(nh),
            sep="",
        )

    def verify(
        i: UInt32,
        j: UInt32,
        k: UInt32,
        t1: UInt32,
        t2: UInt32,
        v: InlineArray[Scalar[T], N],
    ):
        var N = 50
        debug_assert(t1 < N)
        debug_assert(t2 < N)
        if (i != j) and (Scalar[T](v[t1] - v[t2]) <= 0):
            print("for ", i, ":", v[k], " failed ", v[t1], " ", v[t2], sep="")

    h = Hist()
    h4 = Hist4()
    for it in range(5):
        for j in range(N):
            v[j] = Scalar[T](random.random_si64(Int64(rmin), Int64(rmax)))

        if it == 2:
            for j in range(N // 2, N // 2 + N // 4):
                v[j] = 4

        h.zero()
        h4.zero()
        debug_assert(h.size() == 0)
        debug_assert(h4.size() == 0)

        for j in range(N):
            h.count(v[j])
            if j < 2000:
                h4.count(v[j], 2)
            else:
                h4.count(v[j], j % 4)

        debug_assert(h.size() == 0)
        debug_assert(h4.size() == 0)
        h.finalize()
        h4.finalize()

        debug_assert(h.size() == N)
        debug_assert(h4.size() == N)

        for j in range(N):
            h.fill(v[j], j)
            if j < 2000:
                h4.fill(v[j], j, 2)
            else:
                h4.fill(v[j], j, j % 4)

        debug_assert(h.off[0] == 0)
        debug_assert(h4.off[0] == 0)
        debug_assert(h.size() == N)
        debug_assert(h4.size() == N)

        for i in range(Hist.nbins()):
            if h.size(i) == 0:
                continue

            var k = h.begin(i)[]
            debug_assert(k < N)

            var kl = (
                h.bin(max(rmin, v[k] - DELTA))
                .cast[DType.uint32]() if (NBINS != 128) else h.bin(
                    v[k] - Scalar[T](DELTA)
                )
                .cast[DType.uint32]()
            )
            var kh = (
                h.bin(min(rmax, v[k] + DELTA))
                .cast[DType.uint32]() if (NBINS != 128) else h.bin(
                    v[k] + Scalar[T](DELTA)
                )
                .cast[DType.uint32]()
            )

            if NBINS == 128:
                debug_assert(kl != i)
                debug_assert(kh != i)

            if NBINS != 128:
                debug_assert(kl <= i)
                debug_assert(kh >= i)

            j = h.begin(kl)
            end = h.end(kl)
            while j != end:
                verify(i, kl, k, k, j[], v)
                j += 1

            j = h.begin(kh)
            end = h.end(kh)
            while j != end:
                verify(i, kh, k, j[], k, v)
                j += 1

    def ftest(mut tot: Int, k: UInt32):
        debug_assert(k >= 0 and k < N)
        tot += 1

    for j in range(N):
        var b0 = h.bin(v[j])
        var w: Int = 0
        var tot: Int = 0

        Histo.forEachInBins(h, v[j], w, ftest, tot)
        # C++: rtot = h.end(b0) - h.begin(b0), i.e. an element count
        var rtot = Int(h.size(b0.cast[DType.uint32]()))
        debug_assert(tot == rtot)

        w = 1
        tot = 0
        Histo.forEachInBins(h, v[j], w, ftest, tot)
        var bp = b0 + 1
        var bm = b0 - 1

        if bp < Int(h.nbins()):
            rtot += Int(h.size(bp.cast[DType.uint32]()))
        if bm >= 0:
            rtot += Int(h.size(bm.cast[DType.uint32]()))

        debug_assert(tot == rtot)
        w = 2
        tot = 0
        Histo.forEachInBins(h, v[j], w, ftest, tot)
        bp += 1
        bm -= 1

        if bp < Int(h.nbins()):
            rtot += Int(h.size(bp.cast[DType.uint32]()))
        if bm >= 0:
            rtot += Int(h.size(bm.cast[DType.uint32]()))

        debug_assert(tot == rtot)


def main() raises:
    go[DType.int16]()
    go[DType.uint8, 128, 8, 4]()
    go[DType.uint16, 313 // 2, 9, 4]()
