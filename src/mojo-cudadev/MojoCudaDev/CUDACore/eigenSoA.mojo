# Mojo port of CUDACore/eigenSoA.h. Transcribed from mojo-serial's EigenSoA.mojo
# (LayoutTensor-based, distinct from MojoBridge.Matrix -- see Matrix.mojo's
# to_layout_tensor() bridge comment for why these are two independently-ported
# Eigen equivalents). This dialect needs several adjustments vs. mojo-serial's
# version: explicit Self. qualification on struct parameters, size_of[]
# instead of DType.sizeof(), an explicit origin on UnsafePointer, and
# __copyinit__'s source parameter must be named `copy`.
from layout import IntTuple, Layout, LayoutTensor
from std.sys.info import size_of

from MojoCudaDev.MojoBridge.DTypes import Typeable


fn isPowerOf2(v: Int32) -> Bool:
    return v and not (v & (v - 1))


# WARNING: THIS STRUCT IS 128-ALIGNED
struct ScalarSoA[T: DType, S: Int](Copyable, Defaultable, Movable, Sized, Typeable):
    comptime Scalar = Scalar[Self.T]
    var _data: InlineArray[Self.Scalar, Self.S]

    @always_inline
    fn __init__(out self):
        constrained[isPowerOf2(Self.S), "SoA stride not a power of 2"]()
        constrained[Self.S * size_of[Self.Scalar]() % 128 == 0, "SoA size not a multiple of 128"]()
        self._data = InlineArray[Self.Scalar, Self.S](fill=0)

    @always_inline
    fn __init__(out self, var list: InlineArray[Self.Scalar, Self.S]):
        constrained[isPowerOf2(Self.S), "SoA stride not a power of 2"]()
        constrained[Self.S * size_of[Self.Scalar]() % 128 == 0, "SoA size not a multiple of 128"]()
        self._data = list^

    @always_inline
    fn __init__(
        out self, var ptr: UnsafePointer[Self.Scalar, MutAnyOrigin], *, var cp: Bool = False
    ):
        constrained[isPowerOf2(Self.S), "SoA stride not a power of 2"]()
        constrained[Self.S * size_of[Self.Scalar]() % 128 == 0, "SoA size not a multiple of 128"]()

        self._data = InlineArray[Self.Scalar, Self.S](uninitialized=True)

        for i in range(Self.S):
            if cp:
                (self._data.unsafe_ptr() + i).init_pointee_copy((ptr + i)[])
            else:
                (self._data.unsafe_ptr() + i).init_pointee_move((ptr + i).take_pointee())

    @always_inline
    fn __moveinit__(out self, deinit take: Self):
        self._data = take._data^

    @always_inline
    fn __copyinit__(out self, copy: Self):
        self._data = copy._data

    @always_inline
    fn __len__(self) -> Int:
        return Self.S

    @always_inline
    fn data(mut self) -> UnsafePointer[Self.Scalar, MutAnyOrigin]:
        return self._data.unsafe_ptr()

    @always_inline
    fn __getitem__(ref self, i: Int) -> ref [self._data] Self.Scalar:
        return self._data[i]

    @always_inline
    fn __setitem__(mut self, i: Int, v: Self.Scalar):
        self._data[i] = v

    @always_inline
    @staticmethod
    fn dtype() -> String:
        return "ScalarSoA[" + Self.T.__repr__() + ", " + String(Self.S) + "]"


# WARNING: THIS STRUCT IS 128-ALIGNED
struct MatrixSoA[T: DType, R: Int, C: Int, S: Int](
    Copyable, Defaultable, Movable, Sized, Typeable
):
    comptime Scalar = Scalar[Self.T]
    comptime _D = InlineArray[Self.Scalar, Self.S * Self.R * Self.C]
    # stride in C++ is coln, row
    comptime Map = Layout(IntTuple(Self.R, Self.C), IntTuple(Self.S, Self.R * Self.S))
    var _data: Self._D

    @always_inline
    fn __init__(out self):
        constrained[isPowerOf2(Self.S), "SoA stride not a power of 2"]()
        constrained[
            Self.R * Self.C * Self.S * size_of[Self.Scalar]() % 128 == 0,
            "SoA size not a multiple of 128",
        ]()
        self._data = Self._D(fill=0)

    @always_inline
    fn __init__(out self, var list: Self._D):
        constrained[isPowerOf2(Self.S), "SoA stride not a power of 2"]()
        constrained[
            Self.R * Self.C * Self.S * size_of[Self.Scalar]() % 128 == 0,
            "SoA size not a multiple of 128",
        ]()
        self._data = list^

    @always_inline
    fn __init__(
        out self, var ptr: UnsafePointer[Self.Scalar, MutAnyOrigin], *, var cp: Bool = False
    ):
        constrained[isPowerOf2(Self.S), "SoA stride not a power of 2"]()
        constrained[
            Self.R * Self.C * Self.S * size_of[Self.Scalar]() % 128 == 0,
            "SoA size not a multiple of 128",
        ]()

        self._data = Self._D(uninitialized=True)

        for i in range(Self.R * Self.C * Self.S):
            if cp:
                (self._data.unsafe_ptr() + i).init_pointee_copy((ptr + i)[])
            else:
                (self._data.unsafe_ptr() + i).init_pointee_move((ptr + i).take_pointee())

    @always_inline
    fn __moveinit__(out self, deinit take: Self):
        self._data = take._data^

    @always_inline
    fn __copyinit__(out self, copy: Self):
        self._data = copy._data

    @always_inline
    fn __len__(self) -> Int:
        return Self.R * Self.C * Self.S

    @always_inline
    fn __getitem__(mut self, i: Int32) -> LayoutTensor[mut=True, Self.T, Self.Map, MutAnyOrigin]:
        return LayoutTensor[mut=True, Self.T, Self.Map, MutAnyOrigin](
            self._data.unsafe_ptr() + i
        )

    @always_inline
    fn __setitem__(mut self, idx: Int32, val: LayoutTensor):
        var dest_slice = self[idx]

        var rows = val.layout.shape[0].value()
        var colns = val.layout.shape[1].value()
        for i in range(rows):
            for j in range(colns):
                dest_slice[i, j] = rebind[Scalar[Self.T]](val[i, j].cast[Self.T]())

    @always_inline
    fn __setitem__(
        mut self,
        idx: Int32,
        first: LayoutTensor,
        second: LayoutTensor,
    ):
        """Eigen::Map::operator<<."""
        var dest_slice = self[idx]

        var dest_rows = dest_slice.layout.shape[0].value()
        var dest_cols = dest_slice.layout.shape[1].value()
        var first_rows = first.layout.shape[0].value()
        var first_cols = first.layout.shape[1].value()
        var second_rows = second.layout.shape[0].value()
        var second_cols = second.layout.shape[1].value()

        var i_first = 0
        var j_first = 0
        var i_second = 0
        var j_second = 0

        for j_dest in range(dest_cols):
            for i_dest in range(dest_rows):
                if j_first < first_cols:
                    dest_slice[i_dest, j_dest] = rebind[Scalar[Self.T]](
                        first[i_first, j_first].cast[Self.T]()
                    )

                    i_first += 1
                    if i_first == first_rows:
                        i_first = 0
                        j_first += 1

                elif j_second < second_cols:
                    dest_slice[i_dest, j_dest] = rebind[Scalar[Self.T]](
                        second[i_second, j_second].cast[Self.T]()
                    )

                    i_second += 1
                    if i_second == second_rows:
                        i_second = 0
                        j_second += 1
                else:
                    return

    @always_inline
    @staticmethod
    fn dtype() -> String:
        return (
            "MatrixSoA["
            + Self.T.__repr__()
            + ", "
            + String(Self.R)
            + ", "
            + String(Self.C)
            + ", "
            + String(Self.S)
            + "]"
        )
