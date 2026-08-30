from MojoCudaDev.MojoBridge.Matrix import Matrix
from MojoCudaDev.MojoBridge.SymmetricEigen import *
from MojoCudaDev.plugin_PixelTriplets.choleskyInversion import invert33, invert55
from MojoCudaDev.plugin_PixelTriplets.FitUtils import riemannFit

alias F = DType.float64


fn identity_err[n: Int](m: Matrix[F, n, n], inv: Matrix[F, n, n]) -> Float64:
    """max |m@inv - I|, so 0 means inv really is the inverse.

    invertNN only writes the lower triangle (matching the C++), so mirror it
    into the upper triangle first, exactly as the C++ callers do.
    """
    var full = inv
    for i in range(n):
        for j in range(i + 1, n):
            full[i, j] = full[j, i]
    var prod = m @ full
    var worst: Float64 = 0.0
    for i in range(n):
        for j in range(n):
            var want: Float64 = 1.0 if i == j else 0.0
            var e = abs(prod[i, j] - want)
            if e > worst:
                worst = e
    return worst


fn main():
    # symmetric positive definite 3x3 (cholesky needs SPD)
    var m3 = Matrix[F, 3, 3]()
    m3[0, 0] = 4.0
    m3[0, 1] = 1.0
    m3[1, 0] = 1.0
    m3[1, 1] = 3.0
    m3[1, 2] = 1.0
    m3[2, 1] = 1.0
    m3[2, 2] = 2.0
    var i3 = Matrix[F, 3, 3]()
    invert33(m3, i3)
    print("invert33  max|M@inv - I| =", identity_err[3](m3, i3))

    # symmetric positive definite 5x5: diagonally dominant tridiagonal
    var m5 = Matrix[F, 5, 5]()
    for i in range(5):
        m5[i, i] = 5.0
        if i > 0:
            m5[i, i - 1] = 1.0
            m5[i - 1, i] = 1.0
    var i5 = Matrix[F, 5, 5]()
    invert55(m5, i5)
    print("invert55  max|M@inv - I| =", identity_err[5](m5, i5))
