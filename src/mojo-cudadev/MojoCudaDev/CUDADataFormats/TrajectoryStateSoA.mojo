# Mojo port of CUDADataFormats/TrajectoryStateSoAT.h. Transcribed from
# mojo-serial's TrajectoryStateSoA.mojo (LayoutTensor-based, matching
# eigenSoA.mojo).
from layout import Layout, LayoutTensor

from MojoCudaDev.CUDACore.eigenSoA import MatrixSoA
from MojoCudaDev.MojoBridge.DTypes import Float, Typeable


# WARNING: THIS STRUCT IS 128-ALIGNED
struct TrajectoryStateSoA[S: Int](Copyable, Defaultable, Movable, Typeable):
    comptime Vector5f = LayoutTensor[DType.float32, Layout.col_major(5, 1), _]
    comptime Vector15f = LayoutTensor[DType.float32, Layout.col_major(15, 1), _]

    comptime Vector5d = LayoutTensor[DType.float64, Layout.col_major(5, 1), _]
    comptime Matrix5d = LayoutTensor[DType.float64, Layout.col_major(5, 5), _]

    var state: MatrixSoA[DType.float32, 5, 1, Self.S]
    var covariance: MatrixSoA[DType.float32, 15, 1, Self.S]

    fn __init__(out self):
        self.state = MatrixSoA[DType.float32, 5, 1, Self.S]()
        self.covariance = MatrixSoA[DType.float32, 15, 1, Self.S]()

    fn __copyinit__(out self, copy: Self):
        self.state = copy.state.copy()
        self.covariance = copy.covariance.copy()

    fn __moveinit__(out self, deinit take: Self):
        self.state = take.state^
        self.covariance = take.covariance^

    @staticmethod
    fn stride() -> Int:
        return Self.S

    fn copyFromCircle[
        T: DType
    ](
        mut self,
        cp: LayoutTensor[T, Layout.col_major(3, 1), _],
        ccov: LayoutTensor[T, Layout.col_major(3, 3), _],
        lp: LayoutTensor[T, Layout.col_major(2, 1), _],
        lcov: LayoutTensor[T, Layout.col_major(2, 2), _],
        b: Float,
        i: Int32,
    ):
        self.state.__setitem__(i, cp, lp)
        self.state[i][2, 0] *= b
        var cov = self.covariance[i]
        cov[0, 0] = ccov[0, 0].cast[DType.float32]()
        cov[1, 0] = ccov[0, 1].cast[DType.float32]()
        cov[2, 0] = b * ccov[0, 2].cast[DType.float32]()
        cov[4, 0] = 0
        cov[3, 0] = 0
        cov[5, 0] = ccov[1, 1].cast[DType.float32]()
        cov[6, 0] = b * ccov[1, 2].cast[DType.float32]()
        cov[8, 0] = 0
        cov[7, 0] = 0
        cov[9, 0] = b * b * ccov[2, 2].cast[DType.float32]()
        cov[11, 0] = 0
        cov[10, 0] = 0
        cov[12, 0] = lcov[0, 0].cast[DType.float32]()
        cov[13, 0] = lcov[0, 1].cast[DType.float32]()
        cov[14, 0] = lcov[1, 1].cast[DType.float32]()

    fn copyFromDense[
        T: DType
    ](
        mut self,
        v: LayoutTensor[T, Layout.col_major(5, 1), _],
        cov: LayoutTensor[T, Layout.col_major(5, 5), _],
        i: Int32,
    ):
        self.state.__setitem__(i, v)
        var ind: Int = 0

        for j in range(5):
            for k in range(j, 5):
                self.covariance[i][ind, 0] = cov[j, k].cast[DType.float32]()
                ind += 1

    fn copyToDense[
        CT: DType
    ](
        mut self,
        mut v: LayoutTensor[mut=True, ...],
        mut cov: LayoutTensor[mut=True, CT, Layout.col_major(5, 5), ...],
        i: Int32,
    ):
        var wx = self.state[i]
        var rows = wx.layout.shape[0].value()
        var colns = wx.layout.shape[1].value()
        for r in range(rows):
            for c in range(colns):
                v[r, c] = rebind[Scalar[v.dtype]](wx[r, c].cast[v.dtype]())
        var ind: Int = 0

        @parameter
        for j in range(5):
            cov[j, j] = rebind[Scalar[CT]](self.covariance[i][ind, 0].cast[CT]())
            ind += 1

            @parameter
            for k in range(j + 1, 5):
                cov[j, k] = rebind[Scalar[CT]](self.covariance[i][ind, 0].cast[CT]())
                cov[k, j] = cov[j, k]
                ind += 1

    @staticmethod
    fn dtype() -> String:
        return "TrajectoryStateSoA[" + String(Self.S) + "]"
