from MojoCudaDev.MojoBridge.Matrix import Matrix
from MojoCudaDev.MojoBridge.Vector import Vector

alias F = DType.float64
alias M3 = Matrix[F, 3, 3]
alias M32 = Matrix[F, 3, 2]
alias C3 = Matrix[F, 3, 1]

fn check(mut failures: Int, name: String, got: Float64, want: Float64):
    if abs(got - want) < 1e-9:
        print("PASS", name, "=", got)
    else:
        print("FAIL", name, "got", got, "want", want)
        failures += 1


fn main():
    var failures: Int = 0
    # --- construction ---
    var z = M3()  # default: all zeros
    check(failures, "default[1,1]", z[1, 1], 0.0)

    var s = M3(Float64(2.5))  # scalar splat
    check(failures, "scalar splat[2,0]", s[2, 0], 2.5)

    var si = M3(7)  # Int splat
    check(failures, "int splat[0,2]", si[0, 2], 7.0)

    var sl: M3 = 4  # IntLiteral, implicit
    check(failures, "literal splat[1,2]", sl[1, 2], 4.0)

    # --- element write / read, both 2D and flat ---
    var a = M3()
    for i in range(3):
        for j in range(3):
            a[i, j] = Float64(3 * i + j + 1)  # 1..9 row-major
    check(failures, "a[0,0]", a[0, 0], 1.0)
    check(failures, "a[2,2]", a[2, 2], 9.0)
    check(failures, "a flat[5]", a[5], 6.0)

    # --- implicit copy is a real copy, not an alias (the ImplicitlyCopyable change) ---
    var acopy = a
    acopy[0, 0] = -100.0
    check(failures, "implicit copy is deep (src)", a[0, 0], 1.0)
    check(failures, "implicit copy is deep (dst)", acopy[0, 0], -100.0)

    var aexp = a.copy()
    aexp[1, 1] = -50.0
    check(failures, "explicit .copy() is deep (src)", a[1, 1], 5.0)
    check(failures, "explicit .copy() is deep (dst)", aexp[1, 1], -50.0)

    # --- arithmetic (exercises the `var res = self` / `return res` sites) ---
    check(failures, "(a+a)[2,1]", (a + a)[2, 1], 16.0)
    check(failures, "(a-a)[2,1]", (a - a)[2, 1], 0.0)
    # `*` on two matrices is the matrix product (same as @), not elementwise
    check(failures, "(a*a)[1,0]", (a * a)[1, 0], 66.0)
    check(failures, "(a*scalar)[1,0]", (a * Float64(3.0))[1, 0], 12.0)
    check(failures, "(a/2)[2,2]", (a / Float64(2.0))[2, 2], 4.5)
    check(failures, "(-a)[0,1]", (-a)[0, 1], -2.0)
    check(failures, "abs(-a)[0,1]", abs(-a)[0, 1], 2.0)

    var acc = a
    acc += a
    check(failures, "+= [0,0]", acc[0, 0], 2.0)

    # --- rounding family (lines 922-967) ---
    var frac = M3(Float64(2.7))
    check(failures, "__floor__", frac.__floor__()[0, 0], 2.0)
    check(failures, "__ceil__", frac.__ceil__()[0, 0], 3.0)
    check(failures, "__trunc__", frac.__trunc__()[0, 0], 2.0)
    check(failures, "__round__", frac.__round__()[0, 0], 3.0)

    # --- reductions ---
    check(failures, "reduce_add", a.reduce_add(), 45.0)  # 1+..+9
    check(failures, "reduce_max", a.reduce_max(), 9.0)
    check(failures, "reduce_min", a.reduce_min(), 1.0)
    check(failures, "squaredNorm", a.squaredNorm(), 285.0)  # 1+4+9+...+81

    # --- matmul against a known product ---
    # a = [[1,2,3],[4,5,6],[7,8,9]]; a@a = [[30,36,42],[66,81,96],[102,126,150]]
    var p = a @ a
    check(failures, "matmul[0,0]", p[0, 0], 30.0)
    check(failures, "matmul[1,1]", p[1, 1], 81.0)
    check(failures, "matmul[2,2]", p[2, 2], 150.0)

    # --- transpose ---
    check(failures, "transpose[0,2]", a.transpose()[0, 2], 7.0)

    # --- block() / head() (lines 1098 / 1118) ---
    var b = a.block[2, 2](1, 1)  # [[5,6],[8,9]]
    check(failures, "block[0,0]", b[0, 0], 5.0)
    check(failures, "block[1,1]", b[1, 1], 9.0)

    var cv = C3()
    cv[0, 0] = 11.0
    cv[1, 0] = 22.0
    cv[2, 0] = 33.0
    check(failures, "head[2][1,0]", cv.head[2]()[1, 0], 22.0)

    # --- set_block write-back ---
    var sb = M3()
    sb.set_block[2, 2](0, 0, b)
    check(failures, "set_block[1,1]", sb[1, 1], 9.0)

    # --- row/col views ---
    check(failures, "col(1)[2]", a.col(1)[2], 8.0)
    check(failures, "num_rows", Float64(a.num_rows()), 3.0)
    check(failures, "ColsAtCompileTime", Float64(M32.ColsAtCompileTime()), 2.0)

    # --- iteration (exercises _MatIterator) ---
    var itsum: Float64 = 0.0
    for v in a.row_iterator():
        itsum += v
    check(failures, "row_iterator sum", itsum, 45.0)

    # --- determinant / inverse on a well-conditioned matrix ---
    var d = M3()
    d[0, 0] = 2.0
    d[1, 1] = 4.0
    d[2, 2] = 8.0
    check(failures, "det diag", d.det(), 64.0)
    check(failures, "inverse diag[1,1]", d.inverse()[1, 1], 0.25)

    # --- Vector, exercised the same way ---
    var v3 = Vector[F, 3]()
    v3[0] = 1.0
    v3[1] = 2.0
    v3[2] = 3.0
    var v3copy = v3
    v3copy[0] = 99.0
    check(failures, "Vector implicit copy is deep (src)", v3[0], 1.0)
    check(failures, "Vector implicit copy is deep (dst)", v3copy[0], 99.0)
    check(failures, "Vector .copy()", v3.copy()[2], 3.0)
    check(failures, "Vector add", (v3 + v3)[1], 4.0)
    check(failures, "Vector reduce_add", v3.reduce_add(), 6.0)

    # --- Matrix from a Vector row (the fill= constructor sites) ---
    var fromrow = Matrix[F, 4, 3](row=v3)
    check(failures, "row-splat[0,1]", fromrow[0, 1], 2.0)
    check(failures, "row-splat[3,2]", fromrow[3, 2], 3.0)

    print("")
    if failures == 0:
        print("ALL CHECKS PASSED")
    else:
        print("FAILURES:", failures)
