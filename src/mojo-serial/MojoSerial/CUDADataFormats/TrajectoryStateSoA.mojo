from layout import Layout, LayoutTensor, IntTuple

from MojoSerial.CUDACore.EigenSoA import MatrixSoA
from MojoSerial.MojoBridge.DTypes import Float, Double, Typeable
from MojoSerial.MojoBridge.Matrix import Matrix


# WARNING: THIS STRUCT IS 128-ALIGNED
@fieldwise_init
struct TrajectoryStateSoA[S: Int32](Copyable, Defaultable, Movable, Typeable):
    alias Vector5f = LayoutTensor[DType.float32, Layout.col_major(5, 1)]
    alias Vector15f = LayoutTensor[DType.float32, Layout.col_major(15, 1)]

    alias Vector5d = LayoutTensor[DType.float64, Layout.col_major(5, 1)]
    alias Matrix5d = LayoutTensor[DType.float64, Layout.col_major(5, 5)]

    var state: MatrixSoA[DType.float32, 5, 1, Int(S)]
    var covariance: MatrixSoA[DType.float32, 15, 1, Int(S)]

    @always_inline
    fn __init__(out self):
        self.state = MatrixSoA[DType.float32, 5, 1, Int(S)]()
        self.covariance = MatrixSoA[DType.float32, 15, 1, Int(S)]()

    @staticmethod
    @always_inline
    fn stride() -> Int32:
        return S

    @always_inline
    fn copyFromCircle[T: DType](
        mut self,
        cp: LayoutTensor[T, Layout.col_major(3, 1)],
        ccov: LayoutTensor[T, Layout.col_major(3, 3)],
        lp: LayoutTensor[T, Layout.col_major(2, 1)],
        lcov: LayoutTensor[T, Layout.col_major(2, 2)],
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

    @always_inline
    fn copyFromDense(
        mut self,
        v: LayoutTensor[_, Layout.col_major(5, 1)],
        cov: LayoutTensor[_, Layout.col_major(5, 5)],
        i: Int32,
    ):
        self.state.__setitem__(i, v)
        var ind: Int = 0

        @parameter
        for j in range(5):

            @parameter
            for k in range(j, 5):
                self.covariance[i][ind, 0] = cov[j, k].cast[DType.float32]()
                ind += 1

    @always_inline
    fn copyToDense(
        self,
        mut v: LayoutTensor,
        mut cov: LayoutTensor[DType.float32, _],
        i: Int32,
    ):
        var wx = self.state[i]
        var rows = wx.layout.shape[0].value()
        var colns = wx.layout.shape[1].value()
        for i in range(rows):
            for j in range(colns):
                v[i, j] = rebind[Scalar[v.dtype]](wx[i, j].cast[v.dtype]())
        var ind: Int = 0

        @parameter
        for j in range(5):
            cov[j, j] = self.covariance[i][ind, 0]
            ind += 1

            @parameter
            for k in range(j + 1, 5):
                cov[j, k] = self.covariance[i][ind, 0]
                cov[k, j] = cov[j, k]
                ind += 1

    @always_inline
    @staticmethod
    fn dtype() -> String:
        return "TrajectoryStateSoA[" + String(S) + "]"
