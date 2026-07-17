from MojoSerial.MojoBridge.Matrix import Matrix, MatrixXd, VectorXd
from MojoSerial.MojoBridge.DTypes import DType

struct _Rfit_circle_fit:
    var par: Matrix[DType.float64, 3, 1]
    var cov: Matrix[DType.float64, 3, 3]
    var q: Int32
    var chi2: Float32

struct _Rfit_line_fit:
    var par: Matrix[DType.float64, 2, 1]
    var cov: Matrix[DType.float64, 2, 2]
    var chi2: Float64

struct _Rfit_helix_fit:
    var par: Matrix[DType.float64, 5, 1]
    var cov: Matrix[DType.float64, 5, 5]
    var chi2_circle: Float32
    var chi2_line: Float32
    var q: Int32

struct Rfit:

    alias d: Float64 = 1e-4

    alias Vector2d = Matrix[DType.float64, 2, 1]
    alias Vector3d = Matrix[DType.float64, 3, 1]
    alias Vector4d = Matrix[DType.float64, 4, 1]
    alias Vector5d = Matrix[DType.float64, 5, 1]
    alias Vector6f = Matrix[DType.float64, 6, 1]

    alias Matrix2d = Matrix[DType.float64, 2, 2]
    alias Matrix3d = Matrix[DType.float64, 3, 3]
    alias Matrix4d = Matrix[DType.float64, 4, 4]
    alias Matrix5d = Matrix[DType.float64, 5, 5]
    alias Matrix6d = Matrix[DType.float64, 6, 6]

    alias Matrix2x3d = Matrix[DType.float64, 2, 3]
    alias Matrix3f = Matrix[DType.float64, 3, 3]
    alias Vector3f = Matrix[DType.float64, 3, 1]
    alias Vector4f = Matrix[DType.float64, 4, 1]
    
    alias MatrixXd = MatrixXd
    alias VectorXd = VectorXd

    alias MatrixNd[N: Int] = Matrix[DType.float64, N, N]
    alias MatrixNplusONEd[N: Int] = Matrix[DType.float64, N + 1, N + 1]
    alias ArrayNd[N: Int] = Matrix[DType.float64, N, N]
    alias Matrix2Nd[N: Int] = Matrix[DType.float64, 2 * N, 2 * N]
    alias Matrix3Nd[N: Int] = Matrix[DType.float64, 3 * N, 3 * N]
    alias Matrix2xNd[N: Int] = Matrix[DType.float64, 2, N]
    alias Matrix3xNd[N: Int] = Matrix[DType.float64, 3, N]
    alias Array2xNd[N: Int] = Matrix[DType.float64, 2, N]
    alias MatrixNx3d[N: Int] = Matrix[DType.float64, N, 3]
    alias MatrixNx5d[N: Int] = Matrix[DType.float64, N, 5]
    alias VectorNd[N: Int] = Matrix[DType.float64, N, 1]
    alias VectorNplusONEd[N: Int] = Matrix[DType.float64, N + 1, 1]
    alias Vector2Nd[N: Int] = Matrix[DType.float64, 2 * N, 1]
    alias Vector3Nd[N: Int] = Matrix[DType.float64, 3 * N, 1]
    alias RowVectorNd[N: Int] = Matrix[DType.float64, 1, 1]
    alias RowVector2Nd[N: Int] = Matrix[DType.float64, 1, 2 * N]

    alias circle_fit = _Rfit_circle_fit
    alias line_fit = _Rfit_line_fit
    alias helix_fit = _Rfit_helix_fit

    alias u_int = UInt32

    @staticmethod
    fn printIt[C: AnyType, RFIT_DEBUG: Bool = False](m: UnsafePointer[C], prefix: String = ""):
        @parameter
        if RFIT_DEBUG:
            var r: u_int = 0
            while r < m[].num_rows():
                var c: u_int = 0
                while c < m[].cols():
                    print(prefix, "Matrix(", r, ",", c, ") =", m[][r, c])
                    c += 1
                r += 1

    @staticmethod
    fn sqr(a: Float64) -> Float64:
        return a * a

    @staticmethod
    fn cross2D(a: Self.Vector2d, b: Self.Vector2d) -> Float64:
        return a[0] * b[1] - a[1] * b[0]

    @staticmethod
    fn loadCovariance2D[M6xNf: AnyType, M2Nd: AnyType](ge: M6xNf, mut hits_cov: M2Nd):
        var hits_in_fit: UInt32 = M6xNf.ColsAtCompileTime()
        var i: UInt32 = 0
        while i < hits_in_fit:
            var ge_idx = 0
            var j = 0
            var l = 0
            hits_cov[i + j * hits_in_fit, i + l * hits_in_fit] = ge.col(i)[ge_idx]
            ge_idx = 2
            j = 1
            l = 1
            hits_cov[i + j * hits_in_fit, i + l * hits_in_fit] = ge.col(i)[ge_idx]
            ge_idx = 1
            j = 1
            l = 0
            hits_cov[i + l * hits_in_fit, i + j * hits_in_fit] = ge.col(i)[ge_idx]
            hits_cov[i + j * hits_in_fit, i + l * hits_in_fit] = ge.col(i)[ge_idx]
            i += 1

    @staticmethod
    fn loadCovariance[M6xNf: AnyType, M3xNd: AnyType](ge: M6xNf, mut hits_cov: M3xNd):
        var hits_in_fit: UInt32 = M6xNf.ColsAtCompileTime()
        var i: UInt32 = 0
        while i < hits_in_fit:
            var ge_idx = 0
            var j = 0
            var l = 0
            hits_cov[i + j * hits_in_fit, i + l * hits_in_fit] = ge.col(i)[ge_idx]
            ge_idx = 2
            j = 1
            l = 1
            hits_cov[i + j * hits_in_fit, i + l * hits_in_fit] = ge.col(i)[ge_idx]
            ge_idx = 5
            j = 2
            l = 2
            hits_cov[i + j * hits_in_fit, i + l * hits_in_fit] = ge.col(i)[ge_idx]
            ge_idx = 1
            j = 1
            l = 0
            hits_cov[i + l * hits_in_fit, i + j * hits_in_fit] = ge.col(i)[ge_idx]
            hits_cov[i + j * hits_in_fit, i + l * hits_in_fit] = ge.col(i)[ge_idx]
            ge_idx = 3
            j = 2
            l = 0
            hits_cov[i + l * hits_in_fit, i + j * hits_in_fit] = ge.col(i)[ge_idx]
            hits_cov[i + j * hits_in_fit, i + l * hits_in_fit] = ge.col(i)[ge_idx]
            ge_idx = 4
            j = 2
            l = 1
            hits_cov[i + l * hits_in_fit, i + j * hits_in_fit] = ge.col(i)[ge_idx]
            hits_cov[i + j * hits_in_fit, i + l * hits_in_fit] = ge.col(i)[ge_idx]
            i += 1

    @staticmethod
    fn par_uvrtopak(mut circle: Self.circle_fit, B: Float64, error: Bool):
        var par_pak = Vector3d()
        var temp0 = circle.par.head(2).squaredNorm()
        var temp1 = math.sqrt(temp0)
        par_pak[0] = math.atan2(Float64(circle.q) * circle.par[0], -Float64(circle.q) * circle.par[1])
        par_pak[1] = Float64(circle.q) * (temp1 - circle.par[2])
        par_pak[2] = circle.par[2] * B
        if error:
            var temp2 = sqr(circle.par[0]) * 1.0 / temp0
            var temp3 = 1.0 / temp1 * Float64(circle.q)
            var J4 = Matrix3d()
            J4[0, 0] = -circle.par[1] * temp2 * 1.0 / sqr(circle.par[0])
            J4[0, 1] = temp2 * 1.0 / circle.par[0]
            J4[0, 2] = 0.0
            J4[1, 0] = circle.par[0] * temp3
            J4[1, 1] = circle.par[1] * temp3
            J4[1, 2] = -Float64(circle.q)
            J4[2, 0] = 0.0
            J4[2, 1] = 0.0
            J4[2, 2] = B
            circle.cov = J4 * circle.cov * J4.transpose()
        circle.par = par_pak

    @staticmethod
    fn fromCircleToPerigee(mut circle: Self.circle_fit):
        var par_pak = Vector3d()
        var temp0 = circle.par.head(2).squaredNorm()
        var temp1 = math.sqrt(temp0)
        par_pak[0] = math.atan2(Float64(circle.q) * circle.par[0], -Float64(circle.q) * circle.par[1])
        par_pak[1] = Float64(circle.q) * (temp1 - circle.par[2])
        par_pak[2] = Float64(circle.q) / circle.par[2]

        var temp2 = sqr(circle.par[0]) * 1.0 / temp0
        var temp3 = 1.0 / temp1 * Float64(circle.q)
        var J4 = Matrix3d()
        J4[0, 0] = -circle.par[1] * temp2 * 1.0 / sqr(circle.par[0])
        J4[0, 1] = temp2 * 1.0 / circle.par[0]
        J4[0, 2] = 0.0
        J4[1, 0] = circle.par[0] * temp3
        J4[1, 1] = circle.par[1] * temp3
        J4[1, 2] = -Float64(circle.q)
        J4[2, 0] = 0.0
        J4[2, 1] = 0.0
        J4[2, 2] = -Float64(circle.q) / (circle.par[2] * circle.par[2])
        circle.cov = J4 * circle.cov * J4.transpose()

        circle.par = par_pak

    @staticmethod
    fn transformToPerigeePlane[VI5: AnyType, MI5: AnyType, VO5: AnyType, MO5: AnyType](ip: VI5, icov: MI5, mut op: VO5, mut ocov: MO5):
        var sinTheta2 = 1.0 / (1.0 + ip[3] * ip[3])
        var sinTheta = math.sqrt(sinTheta2)
        var cosTheta = ip[3] * sinTheta

        op[0] = sinTheta * ip[2]
        op[1] = 0.0
        op[2] = -ip[3]
        op[3] = ip[1]
        op[4] = -ip[4]

        var J = Matrix5d.Zero()

        J[0, 2] = sinTheta
        J[0, 3] = -sinTheta2 * cosTheta * ip[2]
        J[1, 0] = 1.0
        J[2, 3] = -1.0
        J[3, 1] = 1.0
        J[4, 4] = -1.0

        ocov = J * icov * J.transpose()
