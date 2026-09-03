from std.sys import align_of, is_gpu
from std.bit import pop_count
from std.math import Ceilable, CeilDivable, Floorable, Truncable, sqrt
from std.utils.numerics import max_finite as _max_finite
from std.utils.numerics import max_or_inf as _max_or_inf
from std.utils.numerics import min_finite as _min_finite
from std.utils.numerics import min_or_neg_inf as _min_or_neg_inf
from std.hashlib.hasher import Hasher
from layout import Layout, LayoutTensor

from MojoSerial.MojoBridge.DTypes import Double, Typeable
from MojoSerial.MojoBridge.Vector import Vector


@fieldwise_init
struct _MatIterator[
    mat_mutability: Bool, //,
    W: DType,
    rows: Int,
    colns: Int,
    mat_origin: Origin[mat_mutability],
    forward: Bool = True,
    row_wise: Bool = True,
](Copyable, Iterator, Movable, Typeable):
    comptime mat_type = Matrix[W, rows, colns]
    comptime T = Scalar[W]
    comptime Element = Self.T

    var index: Int
    var src: Pointer[Self.mat_type, mat_origin]

    def __next_ref__(mut self) -> Self.T:
        comptime if forward:
            self.index += 1
            return self.src[][self.index - 1, row_wise]
        else:
            self.index -= 1
            return self.src[][self.index, row_wise]

    @always_inline
    def __next__(mut self) -> Self.T:
        return self.__next_ref__()

    @always_inline
    def __has_next__(self) -> Bool:
        return self.__len__() > 0

    @always_inline
    def __iter__(self) -> Self:
        return self

    def __len__(self) -> Int:
        comptime if forward:
            return len(self.src[]) - self.index
        else:
            return self.index

    @always_inline
    @staticmethod
    def dtype() -> String:
        return (
            "_MatIterator["
            + String(mat_mutability)
            + ", "
            + W.__repr__()
            + ", "
            + String(rows)
            + ", "
            + String(colns)
            + ", Origin["
            + String(mat_mutability)
            + "], "
            + String(row_wise)
            + "]"
        )


# Common interface satisfied by both Matrix (owns its storage) and Map (a
# strided view over external storage), so generic code can accept either.
trait MatrixLike:
    comptime ElemType: DType
    comptime Rows: Int

    def __getitem__(self, i: Int, j: Int) -> Scalar[Self.ElemType]:
        ...

    def __setitem__(mut self, i: Int, j: Int, val: Scalar[Self.ElemType]):
        ...

    def __getitem__(self, i: Int) -> Scalar[Self.ElemType]:
        ...

    def __setitem__(mut self, i: Int, val: Scalar[Self.ElemType]):
        ...

    def num_rows(self) -> Int:
        ...

    def block[
        br: Int, bc: Int
    ](self, row: Int, col: Int) -> Matrix[Self.ElemType, br, bc]:
        ...

    def head[n: Int](self) -> Matrix[Self.ElemType, n, 1]:
        ...

    def col(self, c: Int) -> Vector[Self.ElemType, Self.Rows]:
        ...

    @staticmethod
    def ColsAtCompileTime() -> Int:
        ...


# A comment about this implementation: it is probably the speediest, but arguably not of the best memory efficiency (?)
# Handling rows in a SIMD structure does give rows immense advantage over columns, it also simplfies implementation... but we are still using InlineArray for memory
# TODO: Is implementing a matrix as an inline array of vectors faster or slower than a direct memory implementation using an unsafe pointer?
struct Matrix[T: DType, rows: Int, colns: Int](
    Absable,
    CeilDivable,
    Ceilable,
    Copyable,
    Defaultable,
    Floorable,
    Hashable,
    MatrixLike,
    Movable,
    Representable,
    Roundable,
    Sized,
    Stringable,
    Truncable,
    Typeable,
    Writable,
):
    comptime ElemType = T
    comptime Rows = rows
    comptime _L = List[List[Scalar[T]]]
    comptime _LS = InlineArray[InlineArray[Scalar[T], colns], rows]
    comptime _R = Vector[T, colns]
    comptime _D = Scalar[T]
    comptime _DC = InlineArray[Vector[T, colns], rows]
    comptime _DB = InlineArray[Vector[DType.bool, colns], rows]
    comptime _Mask = Matrix[DType.bool, rows, colns]
    var _data: Self._DC

    # SIMD specifics

    comptime device_type: AnyType = Self

    def _to_device_type(self, target: OpaquePointer):
        target.bitcast[Self.device_type]()[] = self

    @staticmethod
    def get_type_name() -> String:
        return (
            "Matrix[" + repr(T) + ", " + repr(rows) + ", " + repr(colns) + "]"
        )

    @staticmethod
    def get_device_type_name() -> String:
        return Self.get_type_name()

    comptime MAX = Self(_max_or_inf[T]())
    comptime MIN = Self(_min_or_neg_inf[T]())
    comptime MAX_FINITE = Self(_max_finite[T]())
    comptime MIN_FINITE = Self(_min_finite[T]())

    comptime _default_alignment = align_of[Self._D]() if is_gpu() else 1

    @doc_private
    @always_inline("nodebug")
    @implicit
    def __init__(out self, value: __mlir_type.index, /):
        # support MLIR assignment for compatibility purposes
        self._data = Self._DC(value)

    # Lifecycle methods
    @always_inline
    def __init__(out self):
        """Default constructor."""
        self._data = Self._DC(fill=Self._R())

    @always_inline
    def __init__(out self, *, uninitialized: Bool):
        """Default unsafe constructor."""
        self._data = Self._DC(uninitialized=uninitialized)

    @always_inline
    def copy(self) -> Self:
        """Explicitly construct a copy of self."""
        return Self.__copyinit__(self)

    @always_inline
    def __init__[U: DType, //](out self, *, var row: Vector[U, colns]):
        """Initialize a matrix from a Vector row object of the same coln-size, splattered across all rows.
        """
        self._data = Self._DC(Self._R(row))

    @always_inline
    def __init__[U: DType, //](out self, *, var coln: Vector[U, colns]):
        """Initialize a matrix from a Vector coln object of the same row-size, splattered across all columns.
        """
        self._data = Self._DC(uninitialized=True)

        comptime for i in range(rows):

            comptime for j in range(colns):
                self[i, j] = coln[i].cast[T]()

    @always_inline
    def __init__[U: DType, //](out self, val: Scalar[U], /):
        """Initializes a matrix with a scalar.
        The scalar is splatted across all the elements of the matrix."""
        self._data = Self._DC(Self._R(val))

    @always_inline
    def __init__(out self, val: Int, /):
        """Initializes a matrix with a signed integer.
        The signed integer is splatted across all the elements of the matrix."""
        self._data = Self._DC(Self._R(val))

    @always_inline
    def __init__(out self, val: UInt, /):
        """Initializes a matrix with a unsigned integer.
        The unsigned integer is splatted across all the elements of the matrix.
        """
        self._data = Self._DC(Self._R(val))

    @always_inline
    @implicit
    def __init__(out self, val: IntLiteral, /):
        """Initializes a matrix with an integer literal (implicit).
        The integer literal is splatted across all the elements of the matrix.
        """
        self._data = Self._DC(Self._R(val))

    @always_inline
    @implicit
    def __init__(out self, *values: Self._D, __list_literal__: () = ()):
        """Constructs a matrix via a variadic list of values in a literal format (implicit).
        """
        self._data = Self._DC(uninitialized=True)
        for i in range(values.__len__()):
            self[i] = values[i]

    @implicit
    def __init__(out self, mat: Self._L):
        """Constructs a matrix via a matrix list representation (implicit)."""
        self._data = Self._DC(uninitialized=True)
        for i in range(min(rows, mat.__len__())):
            for j in range(min(colns, mat[0].__len__())):
                self[i, j] = mat[i][j]

    @implicit
    def __init__(out self, mat: Self._LS):
        """Constructs a matrix via a matrix inline array representation (implicit).
        """
        self._data = Self._DC(uninitialized=True)

        comptime for i in range(rows):

            comptime for j in range(colns):
                self[i, j] = mat[i][j]

    @implicit
    def __init__(out self, var data: Self._DC):
        """Constructs a matrix via a matrix inline array internal data object (implicit).
        """
        self._data = data^

    def __init__[
        vrows: Int, vcolns: Int, //
    ](out self, mat: Matrix[T, vrows, vcolns]):
        """Initialize a matrix from an arbitrary matrix. Might cause data loss.
        """
        self._data = Self._DC(uninitialized=True)

        comptime for i in range(min(rows * colns, vrows * vcolns)):
            self[i] = mat[i]

    def __init__[U: DType, //](out self, mat: Matrix[U, rows, colns]):
        """Initialize a matrix from a matrix of the same size of a different data type.
        """
        self._data = Self._DC(uninitialized=True)

        comptime for i in range(rows * colns):
            self[i] = mat[i].cast[T]()

    def __init__[
        *, row_offset: Int, coln_offset: Int
    ](out self, mat: Matrix[T, *_]):
        """Initializes a matrix as a slice of another matrix with specified output size and offset.
        """
        self._data = Self._DC(uninitialized=True)
        var u = 0

        comptime for i in range(row_offset, rows + row_offset):
            var v = 0

            comptime for j in range(coln_offset, colns + coln_offset):
                self[u, v] = mat[i, j]
                v += 1
            u += 1

    # Compatibility with V1 Matrices

    def __init__[vsize: Int, //](out self, vec: Vector[T, vsize]):
        """Initialize a matrix from an arbitrary vector (V1 format). Might cause data loss.
        """
        self._data = Self._DC(uninitialized=True)

        comptime for i in range(min(rows * colns, vsize)):
            self[i] = vec[i]

    @implicit
    def __init__(out self, values: List[Self._D], /):
        """Initialize a matrix from a list of values. Might cause data loss."""
        self._data = Self._DC(uninitialized=True)
        for i in range(min(self.__len__(), values.__len__())):
            self[i] = values[i]

    @always_inline
    def __getitem__(self, i: Int) -> Self._D:
        return self._data[i // colns][i % colns]

    @always_inline
    def __getitem__(self, i: Int, row_wise: Bool) -> Self._D:
        if row_wise:
            return self._data[i // colns][i % colns]
        else:
            return self._data[i % rows][i // rows]

    @always_inline
    def __setitem__(mut self, i: Int, val: Self._D):
        self._data[i // colns][i % colns] = val

    @always_inline
    def __setitem__(mut self, i: Int, row_wise: Bool, val: Self._D):
        if row_wise:
            self._data[i // colns][i % colns] = val
        else:
            self._data[i % rows][i // rows] = val

    @always_inline
    def __len__(self) -> Int:
        return rows * colns

    @always_inline
    @staticmethod
    def Zero() -> Self:
        return Self()

    @always_inline
    @staticmethod
    def Constant(val: Self._D) -> Self:
        var res = Self()
        comptime for i in range(rows * colns):
            res[i] = val
        return res

    @staticmethod
    def RowsAtCompileTime() -> Int:
        return rows

    @staticmethod
    def ColsAtCompileTime() -> Int:
        return colns

    @always_inline
    def num_rows(self) -> Int:
        return rows

    @always_inline
    def cols(self) -> Int:
        return colns

    # Operators

    @always_inline
    def __getitem__(self, i: Int, j: Int) -> Self._D:
        return self._data[i][j]

    @always_inline
    def __setitem__(mut self, i: Int, j: Int, val: Self._D):
        self._data[i][j] = val

    def __iter__(ref self) -> _MatIterator[T, rows, colns, origin_of(self)]:
        return _MatIterator[T, rows, colns, origin_of(self)](
            0, Pointer(to=self)
        )

    @always_inline
    def __contains__(self, value: Self._D) -> Bool:
        var res = False

        comptime for i in range(rows):
            res = res and self._data[i].__contains__(value)
            if res:
                return res
        return res

    @always_inline
    def __add__(self, rhs: Self) -> Self:
        comptime assert T.is_numeric(), "DType must be numeric"
        var res = self

        comptime for i in range(rows):
            res._data[i] = res._data[i] + rhs._data[i]
        return res

    @always_inline
    def __sub__(self, rhs: Self) -> Self:
        comptime assert T.is_numeric(), "DType must be numeric"
        var res = self

        comptime for i in range(rows):
            res._data[i] = res._data[i] - rhs._data[i]
        return res

    @always_inline
    def __mul__[trp: Int, //](self, rhs: Matrix[T, colns, trp]) -> Matrix[T, rows, trp]:
        """Matrix product (rows x colns) @ (colns x trp), matching Eigen's operator*."""
        return self @ rhs

    @always_inline
    def __mul__(self, scalar: Self._D) -> Self:
        comptime assert T.is_numeric(), "DType must be numeric"
        var res = self
        # Splat explicitly: relying on implicit Scalar->Vector conversion in
        # `res._data[i] * scalar` silently only fills lane 0 in this Mojo build.
        var splatted = Self._R(scalar)

        comptime for i in range(rows):
            res._data[i] = res._data[i] * splatted
        return res

    @always_inline
    def __rmul__(self, scalar: Self._D) -> Self:
        return self * scalar

    @no_inline
    def __matmul__[
        trp: Int, //
    ](self, rhs: Matrix[T, colns, trp]) -> Matrix[T, rows, trp]:
        var res = Matrix[T, rows, trp]()

        comptime for i in range(rows):

            comptime for j in range(trp):
                res[i, j] = self._row_by_coln(rhs, i, j)
        return res

    @always_inline
    def __truediv__(self, rhs: Self) -> Self:
        comptime assert T.is_numeric(), "DType must be numeric"
        var res = self

        comptime for i in range(rows):
            res._data[i] = res._data[i] / rhs._data[i]
        return res

    @always_inline
    def __truediv__(self, scalar: Self._D) -> Self:
        comptime assert T.is_numeric(), "DType must be numeric"
        var res = self
        var splatted = Self._R(scalar)

        comptime for i in range(rows):
            res._data[i] = res._data[i] / splatted
        return res

    @always_inline
    def __floordiv__(self, rhs: Self) -> Self:
        comptime assert T.is_numeric(), "DType must be numeric"
        var res = self

        comptime for i in range(rows):
            res._data[i] = res._data[i] // rhs._data[i]
        return res

    @always_inline
    def __mod__(self, rhs: Self) -> Self:
        comptime assert T.is_numeric(), "DType must be numeric"
        var res = self

        comptime for i in range(rows):
            res._data[i] = res._data[i] % rhs._data[i]
        return res

    @no_inline
    def __pow__(
        self: Matrix[T, rows, colns], exp: Int
    ) -> Matrix[T, rows, colns]:
        comptime assert (
            rows == colns
        ), "Can only calculate power of a square matrix"
        comptime sq = Matrix[T, rows, rows]

        if exp < 0:
            return ~(self**-exp)
        elif exp == 0:
            return Matrix[T, rows, colns].identity()
        elif exp == 1:
            return self
        var res: Matrix[T, rows, colns] = self
        for _ in range(2, exp + 1):
            res = rebind[sq](self) @ res
        return res

    @always_inline
    def __lt__(self, rhs: Self) -> Self._Mask:
        var res = Self._Mask()

        comptime for i in range(rows):
            res._data[i] = self._data[i] < rhs._data[i]
        return res

    @always_inline
    def __le__(self, rhs: Self) -> Self._Mask:
        var res = Self._Mask()

        comptime for i in range(rows):
            res._data[i] = self._data[i] <= rhs._data[i]
        return res

    @always_inline
    def __eq__(self, rhs: Self) -> Self._Mask:
        var res = Self._Mask()

        comptime for i in range(rows):
            res._data[i] = self._data[i] == rhs._data[i]
        return res

    @always_inline
    def __ne__(self, rhs: Self) -> Self._Mask:
        var res = Self._Mask()

        comptime for i in range(rows):
            res._data[i] = self._data[i] != rhs._data[i]
        return res

    @always_inline
    def __gt__(self, rhs: Self) -> Self._Mask:
        var res = Self._Mask()

        comptime for i in range(rows):
            res._data[i] = self._data[i] > rhs._data[i]
        return res

    @always_inline
    def __ge__(self, rhs: Self) -> Self._Mask:
        var res = Self._Mask()

        comptime for i in range(rows):
            res._data[i] = self._data[i] >= rhs._data[i]
        return res

    @always_inline
    def __pos__(self) -> Self:
        comptime assert T.is_numeric(), "DType must be numeric"
        return self

    @always_inline
    def __neg__(self) -> Self:
        comptime assert T.is_numeric(), "DType must be numeric"
        var res = self

        comptime for i in range(rows):
            res[i] = -res[i]
        return res

    @always_inline
    def __and__(self, rhs: Self) -> Self:
        comptime assert (
            T.is_integral() or T == DType.bool
        ), "DType must be an integral or bool type"
        var res = self

        comptime for i in range(rows):
            res._data[i] = res._data[i] & rhs._data[i]
        return res

    @always_inline
    def __xor__(self, rhs: Self) -> Self:
        comptime assert (
            T.is_integral() or T == DType.bool
        ), "DType must be an integral or bool type"
        var res = self

        comptime for i in range(rows):
            res._data[i] = res._data[i] ^ rhs._data[i]
        return res

    @always_inline
    def __or__(self, rhs: Self) -> Self:
        comptime assert (
            T.is_integral() or T == DType.bool
        ), "DType must be an integral or bool type"
        var res = self

        comptime for i in range(rows):
            res._data[i] = res._data[i] | rhs._data[i]
        return res

    @always_inline
    def __lshift__(self, rhs: Self) -> Self:
        comptime assert T.is_integral(), "DType must be an integral type"
        var res = self

        comptime for i in range(rows):
            res._data[i] = res._data[i] << rhs._data[i]
        return res

    @always_inline
    def __rshift__(self, rhs: Self) -> Self:
        comptime assert T.is_integral(), "DType must be an integral type"
        var res = self

        comptime for i in range(rows):
            res._data[i] = res._data[i] >> rhs._data[i]
        return res

    @no_inline
    def __invert__[
        W: DType, *, protect: Bool = False
    ](self: Matrix[T, rows, colns]) -> Matrix[W, rows, colns]:
        comptime assert (
            rows == colns
        ), "Can only find inverse of a square matrix"
        debug_assert(
            abs(self.det[DType.float64]()) > 1e-9, "Matrix is not invertible"
        )
        # if this assert fails, we'll return a weird value
        comptime n = rows

        var mat = self.cast[DType.float64]()
        var idn = Matrix[DType.float64, rows, colns].identity()

        comptime for i in range(n):
            var pivot = i

            comptime for j in range(i, n):
                if abs(mat[j, i]) > abs(mat[pivot, i]):
                    pivot = j

            mat._data[i], mat._data[pivot] = mat._data[pivot], mat._data[i]
            idn._data[i], idn._data[pivot] = idn._data[pivot], idn._data[i]

            if abs(mat[i, i]) < 1e-9:
                return idn.cast[W]()

            var div = mat[i, i]

            comptime for j in range(n):
                mat[i, j] /= div
                idn[i, j] /= div

            comptime for j in range(n):
                if i != j:
                    var mult = mat[j, i]

                    comptime for k in range(n):
                        mat[j, k] -= mult * mat[i, k]
                        idn[j, k] -= mult * idn[i, k]

        comptime if protect:

            comptime if W in (
                DType.uint8,
                DType.uint16,
                DType.uint32,
                DType.uint64,
                DType.uint128,
                DType.uint256,
            ):

                comptime for i in range(n * n):
                    if idn[i] < 1e-9:
                        idn[i] = 0
        return idn.cast[W]()

    @always_inline
    def __invert__(self) -> Self:
        return self.__invert__[T]()

    # In place operations

    @always_inline("nodebug")
    def __iadd__(mut self, rhs: Self):
        comptime assert T.is_numeric(), "DType must be numeric"
        self = self + rhs

    @always_inline("nodebug")
    def __isub__(mut self, rhs: Self):
        comptime assert T.is_numeric(), "DType must be numeric"
        self = self - rhs

    @always_inline("nodebug")
    def __imul__(mut self, rhs: Self):
        comptime assert T.is_numeric(), "DType must be numeric"
        comptime assert (
            rows == colns
        ), "In-place matrix multiply requires a square matrix"
        var result = rebind[Matrix[T, colns, colns]](self) @ rebind[Matrix[T, colns, colns]](rhs)
        self = rebind[Self](result)

    @always_inline("nodebug")
    def __imul__(mut self, scalar: Self._D):
        comptime assert T.is_numeric(), "DType must be numeric"
        self = self * scalar

    @always_inline("nodebug")
    def __itruediv__(mut self, rhs: Self):
        comptime assert T.is_numeric(), "DType must be numeric"
        self = self / rhs

    @always_inline("nodebug")
    def __itruediv__(mut self, scalar: Self._D):
        comptime assert T.is_numeric(), "DType must be numeric"
        self = self / scalar

    @always_inline("nodebug")
    def __ifloordiv__(mut self, rhs: Self):
        comptime assert T.is_numeric(), "DType must be numeric"
        self = self // rhs

    @always_inline("nodebug")
    def __imod__(mut self, rhs: Self):
        comptime assert T.is_numeric(), "DType must be numeric"
        self = self.__mod__(rhs)

    @always_inline("nodebug")
    def __ipow__(mut self: Matrix[T, rows, colns], rhs: Int):
        comptime assert T.is_numeric(), "DType must be numeric"
        comptime assert (
            rows == colns
        ), "Can only calculate power of a square matrix"
        self = self.__pow__(rhs)

    @always_inline("nodebug")
    def __iand__(mut self, rhs: Self):
        comptime assert (
            T.is_integral() or T == DType.bool
        ), "DType must be an integral or bool type"
        self = self & rhs

    @always_inline("nodebug")
    def __ixor__(mut self, rhs: Self):
        comptime assert (
            T.is_integral() or T == DType.bool
        ), "DType must be an integral or bool type"
        self = self ^ rhs

    @always_inline("nodebug")
    def __ior__(mut self, rhs: Self):
        comptime assert (
            T.is_integral() or T == DType.bool
        ), "DType must be an integral or bool type"
        self = self | rhs

    @always_inline("nodebug")
    def __ilshift__(mut self, rhs: Self):
        comptime assert T.is_integral(), "DType must be an integral type"
        self = self << rhs

    @always_inline("nodebug")
    def __irshift__(mut self, rhs: Self):
        comptime assert T.is_integral(), "DType must be an integral type"
        self = self >> rhs

    @always_inline("nodebug")
    def __iinvert__(mut self):
        comptime assert (
            T.is_integral() or T == DType.bool
        ), "DType must be an integral or bool type"
        self = ~self

    # Reversed operations

    @always_inline
    def __radd__(self, value: Self) -> Self:
        comptime assert T.is_numeric(), "DType must be numeric"
        return value + self

    @always_inline
    def __rsub__(self, value: Self) -> Self:
        comptime assert T.is_numeric(), "DType must be numeric"
        return value - self

    @always_inline
    def __rfloordiv__(self, rhs: Self) -> Self:
        comptime assert T.is_numeric(), "DType must be numeric"
        return rhs // self

    @always_inline
    def __rtruediv__(self, value: Self) -> Self:
        comptime assert T.is_numeric(), "DType must be numeric"
        return value / self

    @always_inline
    def __rmod__(self, value: Self) -> Self:
        comptime assert T.is_numeric(), "DType must be numeric"
        return value % self

    @always_inline
    def __rand__(self, value: Self) -> Self:
        comptime assert (
            T.is_integral() or T == DType.bool
        ), "DType be an integral or bool type"
        return value & self

    @always_inline
    def __rxor__(self, value: Self) -> Self:
        comptime assert (
            T.is_integral() or T == DType.bool
        ), "DType be an integral or bool type"
        return value ^ self

    @always_inline
    def __ror__(self, value: Self) -> Self:
        comptime assert (
            T.is_integral() or T == DType.bool
        ), "DType be an integral or bool type"
        return value | self

    @always_inline
    def __rlshift__(self, value: Self) -> Self:
        comptime assert T.is_integral(), "DType be an integral type"
        return value << self

    @always_inline
    def __rrshift__(self, value: Self) -> Self:
        comptime assert T.is_integral(), "DType be an integral type"
        return value >> self

    # Trait conformance

    @always_inline
    @staticmethod
    def dtype() -> String:
        return (
            "Matrix["
            + T.__repr__()
            + ", "
            + String(rows)
            + ", "
            + String(colns)
            + "]"
        )

    @always_inline
    def __str__(self) -> String:
        return String.write(self)

    @no_inline
    def __repr__(self) -> String:
        var output = String()
        output.write("Matrix[" + T.__repr__() + ", ", rows, ", ", colns, "](")
        for i in range(self.__len__()):
            output.write(self[i])
            if i < self.__len__() - 1:
                output.write(", ")
        output.write(")")
        return output^

    @always_inline
    def __floor__(self) -> Self:
        var res = Self()

        comptime for i in range(rows):
            res._data[i] = self._data[i].__floor__()
        return res

    @always_inline
    def __ceil__(self) -> Self:
        var res = Self()

        comptime for i in range(rows):
            res._data[i] = self._data[i].__ceil__()
        return res

    @always_inline
    def __trunc__(self) -> Self:
        var res = Self()

        comptime for i in range(rows):
            res._data[i] = self._data[i].__trunc__()
        return res

    @always_inline
    def __abs__(self) -> Self:
        var res = Self()

        comptime for i in range(rows):
            res._data[i] = self._data[i].__abs__()
        return res

    @always_inline
    def __round__(self) -> Self:
        var res = Self()

        comptime for i in range(rows):
            res._data[i] = self._data[i].__round__()
        return res

    @always_inline
    def __round__(self, ndigits: Int) -> Self:
        var res = Self()

        comptime for i in range(rows):
            res._data[i] = self._data[i].__round__(ndigits)
        return res

    @always_inline
    def __ceildiv__(self, denominator: Self) -> Self:
        return self.__truediv__(denominator).__round__()

    def __hash__[H: Hasher](self, mut hasher: H):
        comptime for i in range(rows):
            self._data[i].__hash__[H](hasher)
        hasher._update_with_simd(Scalar[DType.uint64](37))

    # Methods

    @always_inline("nodebug")
    def _refine[
        T: DType = Self.T, rows: Int = Self.rows, colns: Int = Self.colns
    ](self) -> Matrix[T, rows, colns]:
        return rebind[Matrix[T, rows, colns]](self)

    @always_inline
    def cast[target: DType](self) -> Matrix[target, rows, colns]:
        comptime if T is target:
            return self._refine[target]()

        comptime if T in (DType.float8_e4m3fn, DType.float8_e5m2):
            comptime assert (
                target
                in (
                    DType.bfloat16,
                    DType.float16,
                    DType.float32,
                    DType.float64,
                )
            ), String(
                (
                    "Only FP8->F64, FP8->F32, FP8->F16, and FP8->BF16"
                    " castings are implemented. "
                ),
                T,
                "->",
                target,
            )

        # low level manip for efficiency
        var res = InlineArray[Vector[target, colns], rows](uninitialized=True)

        comptime for i in range(rows):
            res[i] = self._data[i].cast[target]()
        return res

    @always_inline
    def is_power_of_two(self) -> Self._Mask:
        comptime assert T.is_integral(), "DType must be integral"
        if T.is_unsigned():
            return self.pop_count() == 1
        else:
            return (self > 0) & (self & (self - 1) == 0)

    @no_inline
    def write_to[W: Writer](self, mut writer: W):
        writer.write("[")

        var width = 0

        comptime for i in range(rows * colns):
            width = max(width, self[i].__str__().__len__())

        comptime for i in range(rows):
            if i != 0:
                writer.write(" ")
            writer.write("[")

            comptime for j in range(colns):
                var _c = width - self[i, j].__str__().__len__()
                writer.write(
                    " " * (_c if _c > 0 else 0)
                    + self[i, j].__str__()
                    + (" " if j < colns - 1 else "")
                )
            writer.write("]")
            if i < rows - 1:
                writer.write("\n")
        writer.write("]")

    def row_iterator(
        ref self,
    ) -> _MatIterator[T, rows, colns, origin_of(self)]:
        return _MatIterator[T, rows, colns, origin_of(self)](
            0, Pointer(to=self)
        )

    def coln_iterator(
        ref self,
    ) -> _MatIterator[T, rows, colns, origin_of(self), row_wise=False]:
        return _MatIterator[T, rows, colns, origin_of(self), row_wise=False](
            0, Pointer(to=self)
        )

    @always_inline
    def row(self, i: Int) -> Vector[T, colns]:
        return self._data[i]

    @no_inline
    def coln(self, j: Int) -> Vector[T, rows]:
        var res = Vector[T, rows]()

        comptime for i in range(rows):
            res[i] = self[i, j]
        return res

    @always_inline
    def col(self, c: Int) -> Vector[T, rows]:
        return self.coln(c)

    @always_inline
    def block[br: Int, bc: Int](self, row: Int, col: Int) -> Matrix[T, br, bc]:
        var res = Matrix[T, br, bc]()
        for r in range(br):
            for c in range(bc):
                res[r, c] = self[row + r, col + c]
        return res

    @always_inline
    def set_block[
        br: Int, bc: Int
    ](mut self, row: Int, col: Int, val: Matrix[T, br, bc]):
        """Writes val into self at (row, col), the write-back counterpart to
        block() (which returns a copy, unlike Eigen's reference-semantics
        .block())."""
        for r in range(br):
            for c in range(bc):
                self[row + r, col + c] = val[r, c]

    #this is not flattening it is just taking first col
    @always_inline
    def head[n: Int](self) -> Matrix[T, n, 1]:
        comptime assert colns == 1, "head() requires a column vector"
        var res = Matrix[T, n, 1]()
        for i in range(n):
            res[i, 0] = self[i, 0]
        return res

    @always_inline
    def squaredNorm(self) -> Self._D:
        var acc: Self._D = 0
        for i in range(rows):
            for j in range(colns):
                acc += self[i, j] * self[i, j]
        return acc

    @always_inline
    def norm(self) -> Self._D:
        return sqrt(self.squaredNorm())

    @always_inline
    def dot(self, other: Self) -> Self._D:
        var acc: Self._D = 0
        for i in range(rows):
            for j in range(colns):
                acc += self[i, j] * other[i, j]
        return acc

    @no_inline
    def _row_by_coln(
        self, other: Matrix[T, colns, _], row: Int, coln: Int
    ) -> Self._D:
        comptime assert T.is_numeric(), "DType must be numeric"
        var sum: Self._D = 0

        comptime for i in range(colns):
            sum += self[row, i] * other[i, coln]
        return sum

    @no_inline
    def transpose(self) -> Matrix[T, colns, rows]:
        var res = Matrix[T, colns, rows]()

        comptime for i in range(colns):

            comptime for j in range(rows):
                res[i, j] = self[j, i]
        return res

    @staticmethod
    @no_inline
    def identity() -> Self:
        comptime assert rows == colns, "Identity can only be a square matrix"
        var res = Self()
        for i in range(rows):
            res[i, i] = 1
        return res

    @always_inline
    def inverse[
        W: DType, *, protect: Bool = False
    ](self: Matrix[T, rows, colns]) -> Matrix[W, rows, colns]:
        comptime assert (
            rows == colns
        ), "Can only find inverse of a square matrix"
        return self.__invert__[W, protect=protect]()

    @always_inline
    def inverse(self) -> Self:
        comptime assert (
            rows == colns
        ), "Can only find inverse of a square matrix"
        return ~self

    @no_inline
    def det[
        W: DType, *, protect: Bool = False
    ](self: Matrix[T, rows, colns]) -> Scalar[W]:
        comptime assert (
            rows == colns
        ), "Can only calculate determinant for a square matrix"
        comptime n = rows

        var mat = self.cast[DType.float64]()
        var det: Double = 1.0

        comptime for i in range(n):
            var pivot = i

            comptime for j in range(i + 1, n):
                if abs(mat[j, i]) > abs(mat[pivot, i]):
                    pivot = j
            if pivot != i:
                mat._data[i], mat._data[pivot] = mat._data[pivot], mat._data[i]
                det *= -1
            if mat[i, i] == 0:
                return 0
            det *= mat[i, i]

            comptime for j in range(i + 1, n):
                var factor: Double = mat[j, i] / mat[i, i]

                comptime for k in range(i + 1, n):
                    mat[j, k] -= factor * mat[i, k]

        comptime if protect:

            comptime if W in (
                DType.uint8,
                DType.uint16,
                DType.uint32,
                DType.uint64,
                DType.uint128,
                DType.uint256,
            ):
                if det < 1e-9:
                    det = 0
        return det.cast[W]()

    @always_inline
    def det[
        *, protect: Bool = False
    ](self: Matrix[T, rows, colns],) -> Self._D:
        return self.det[T, protect=protect]()

    @always_inline
    def clamp(self, lower_bound: Self, upper_bound: Self) -> Self:
        var res = self

        comptime for i in range(rows):
            res._data[i] = res._data[i].clamp(
                lower_bound._data[i], upper_bound._data[i]
            )
        return res

    @always_inline
    def fma(self, multiplier: Self, accumulator: Self) -> Self:
        comptime assert T.is_numeric(), "DType must be numeric"
        var res = self
        for i in range(rows):
            res._data[i] = res._data[i].fma(
                multiplier._data[i], accumulator._data[i]
            )
        return res

    def slice[
        output_rows: Int,
        output_colns: Int,
        /,
        *,
        row_offset: Int = 0,
        coln_offset: Int = 0,
    ](self) -> Matrix[T, output_rows, output_colns]:
        comptime assert (
            0 <= row_offset < output_rows + row_offset <= rows
        ), "Output rows must be a positive integer less than rows"
        comptime assert (
            0 <= coln_offset < output_colns + coln_offset <= rows
        ), "Output colns must be a positive integer less than colns"

        comptime if output_rows == 1 and output_colns == 1:
            return self[row_offset, coln_offset]

        return Matrix[T, output_rows, output_colns].__init__[
            row_offset=row_offset, coln_offset=coln_offset
        ](self)

    def insert[
        *, row_offset: Int = 0, coln_offset: Int = 0
    ](self, mat: Matrix[T, *_]) -> Self:
        comptime input_rows = mat.rows
        comptime input_colns = mat.colns
        comptime assert (
            0 <= row_offset < input_rows + row_offset <= rows
        ), "Insertion position must not exceed the rows of the matrix"
        comptime assert (
            0 <= coln_offset < input_colns + coln_offset <= rows
        ), "Insertion position must not exceed the colns of the matrix"

        comptime if rows == 1 and colns == 1:
            comptime assert (
                input_rows == 1 and input_colns == 1
            ), "The input width must be 1 if the size of the matrix is 1"
            return mat[0]

        var res = self

        comptime for i in range(row_offset, rows):
            res._data[i] = res._data[i].insert[offset=coln_offset](mat._data[i])
        return res

    def iinsert[
        *, row_offset: Int = 0, coln_offset: Int = 0
    ](mut self, mat: Matrix[T, *_]):
        comptime input_rows = mat.rows
        comptime input_colns = mat.colns
        comptime assert (
            0 <= row_offset < input_rows + row_offset <= rows
        ), "Insertion position must not exceed the rows of the matrix"
        comptime assert (
            0 <= coln_offset < input_colns + coln_offset <= rows
        ), "Insertion position must not exceed the colns of the matrix"

        comptime if rows == 1 and colns == 1:
            comptime assert (
                input_rows == 1 and input_colns == 1
            ), "The input width must be 1 if the size of the matrix is 1"
            self[0] = mat[0]

        comptime for i in range(row_offset, rows):
            self._data[i] = self._data[i].insert[offset=coln_offset](
                mat._data[i]
            )

    def row_stack[
        mcolns: Int, //
    ](self, other: Matrix[T, rows, mcolns]) -> Matrix[T, rows, colns + mcolns]:
        var res = Matrix[T, rows, colns + mcolns](uninitialized=True)

        comptime for i in range(rows):
            res._data[i] = self._data[i].join(other._data[i])
        return res

    def coln_stack[
        mrows: Int, //
    ](self, other: Matrix[T, mrows, colns]) -> Matrix[T, rows + mrows, colns]:
        var res = Matrix[T, rows + mrows, colns](uninitialized=True)

        comptime for i in range(rows):
            res._data[i] = self._data[i]

        comptime for i in range(mrows):
            res._data[rows + i] = other._data[i]
        return res

    def split[
        factor: Int = 2
    ](self) -> InlineArray[
        Matrix[T, rows // factor, colns // factor], factor * factor
    ]:
        comptime assert (
            rows == colns and rows % factor == 0
        ), "Can only do integral splits on square matrices"
        var res = InlineArray[
            Matrix[T, rows // factor, colns // factor], factor * factor
        ](uninitialized=True)
        var i = 0

        comptime for row_offset in range(0, rows, rows // factor):

            comptime for coln_offset in range(0, colns, colns // factor):
                res[i] = self.slice[
                    rows // factor,
                    colns // factor,
                    row_offset=row_offset,
                    coln_offset=coln_offset,
                ]()
                i += 1
        return res

    # Reductions

    def reduce_max(self) -> Self._D:
        comptime if rows == 1 and colns == 1:
            return self[0]
        var A = self._data[0].reduce_max()

        comptime for i in range(rows):
            A = max(A, self._data[i].reduce_max())
        return A

    def reduce_min(self) -> Self._D:
        comptime if rows == 1 and colns == 1:
            return self[0]
        var A = self._data[0].reduce_min()

        comptime for i in range(rows):
            A = min(A, self._data[i].reduce_min())
        return A

    def reduce_add(self) -> Self._D:
        comptime if rows == 1 and colns == 1:
            return self[0]
        var A = self._data[0].reduce_add()

        comptime for i in range(1, rows):
            A = A + self._data[i].reduce_add()
        return A

    def reduce_mul(self) -> Self._D:
        comptime if rows == 1 and colns == 1:
            return self[0]
        var A = self._data[0].reduce_mul()

        comptime for i in range(1, rows):
            A = A * self._data[i].reduce_mul()
        return A

    def reduce_and(self) -> Self._D:
        comptime if rows == 1 and colns == 1:
            return self[0]
        var A = self._data[0].reduce_and()

        comptime for i in range(rows):
            A = A & self._data[i].reduce_and()
        return A

    def reduce_or(self) -> Self._D:
        comptime if rows == 1 and colns == 1:
            return self[0]
        var A = self._data[0].reduce_or()

        comptime for i in range(rows):
            A = A | self._data[i].reduce_or()
        return A

    def reduce_bit_count(self) -> Int:
        comptime assert (
            T.is_integral() or T == DType.bool
        ), "Expected either integral or bool type"

        comptime if T == DType.bool:
            return Int(self.cast[DType.uint8]().reduce_add())
        else:
            return Int(self.pop_count().reduce_add())

    def pop_count(self) -> Self:
        var res = Self._DC(uninitialized=True)

        comptime for i in range(rows):
            res[i] = Vector[T, colns](pop_count(self._data[i]._data))
        return Self(res)


struct Map[T: DType, rows: Int, colns: Int, default_inner_stride: Int = 1](
    Copyable,
    MatrixLike,
    Movable,
    Typeable,
):
    comptime ElemType = Self.T
    comptime Rows = Self.rows
    var data: UnsafePointer[Scalar[Self.T]]
    var inner_stride: Int
    var outer_stride: Int

    @always_inline
    def __init__(out self, ptr: UnsafePointer[Scalar[Self.T]]):
        self.data = ptr
        self.inner_stride = default_inner_stride
        self.outer_stride = rows * default_inner_stride

    @always_inline
    def __init__(out self, ptr: UnsafePointer[Scalar[Self.T]], inner_stride: Int):
        self.data = ptr
        self.inner_stride = inner_stride
        self.outer_stride = rows * inner_stride

    @always_inline
    def __init__(
        out self,
        ptr: UnsafePointer[Scalar[Self.T]],
        inner_stride: Int,
        outer_stride: Int,
    ):
        self.data = ptr
        self.inner_stride = inner_stride
        self.outer_stride = outer_stride

    @always_inline
    def __getitem__(self, r: Int, c: Int) -> Scalar[Self.T]:
        return (
            self.data + c * self.outer_stride + r * self.inner_stride
        )[]

    @always_inline
    def __setitem__(mut self, r: Int, c: Int, val: Scalar[Self.T]):
        (self.data + c * self.outer_stride + r * self.inner_stride)[] = val
    #this is not flattening it is just taking first col
    @always_inline
    def __getitem__(self, i: Int) -> Scalar[Self.T]:
        comptime assert colns == 1, "Map 1D access requires a column vector"
        return self[i, 0]

    @always_inline
    def __setitem__(mut self, i: Int, val: Scalar[Self.T]):
        comptime assert colns == 1, "Map 1D access requires a column vector"
        self[i, 0] = val

    @always_inline
    def col(self, c: Int) -> Vector[T, rows]:
        var res = Vector[T, rows]()
        for r in range(rows):
            res[r] = self[r, c]
        return res

    @always_inline
    def block[br: Int, bc: Int](self, row: Int, col: Int) -> Matrix[T, br, bc]:
        var res = Matrix[T, br, bc]()
        for r in range(br):
            for c in range(bc):
                res[r, c] = self[row + r, col + c]
        return res

    #this is not flattening it is just taking first col
    @always_inline
    def head[n: Int](self) -> Matrix[T, n, 1]:
        comptime assert colns == 1, "Map head() requires a column vector"
        var res = Matrix[T, n, 1]()
        for i in range(n):
            res[i, 0] = self[i, 0]
        return res

    @staticmethod
    def RowsAtCompileTime() -> Int:
        return Self.rows

    @staticmethod
    def ColsAtCompileTime() -> Int:
        return Self.colns

    @always_inline
    def num_rows(self) -> Int:
        return Self.rows

    @always_inline
    @staticmethod
    def dtype() -> String:
        return (
            "Map["
            + Self.T.__repr__()
            + ", "
            + String(Self.rows)
            + ", "
            + String(Self.colns)
            + "]"
        )


@fieldwise_init
struct MatrixXd(Defaultable, Movable, Typeable):
    var _rows: Int
    var _cols: Int
    var _data: List[Float64]

    @always_inline
    def __init__(out self):
        self._rows = 0
        self._cols = 0
        self._data = List[Float64]()

    @always_inline
    def __init__(out self, rows: Int, cols: Int, val: Float64 = 0.0):
        self._rows = rows
        self._cols = cols
        self._data = List[Float64](length=rows * cols, fill=val)

    @always_inline
    def rows(self) -> Int:
        return self._rows

    @always_inline
    def cols(self) -> Int:
        return self._cols

    @always_inline
    def __getitem__(self, r: Int, c: Int) -> Float64:
        return self._data[r + c * self._rows]

    @always_inline
    def __setitem__(mut self, r: Int, c: Int, val: Float64):
        self._data[r + c * self._rows] = val

    @staticmethod
    def Zero[rows: Int, cols: Int](
        rows_hint: IntLiteral, cols_hint: IntLiteral
    ) -> Matrix[DType.float64, rows, cols]:
        return Matrix[DType.float64, rows, cols]()

    @staticmethod
    def zero[rows: Int, cols: Int](
        rows_hint: IntLiteral, cols_hint: IntLiteral
    ) -> Matrix[DType.float64, rows, cols]:
        return Matrix[DType.float64, rows, cols]()



    @staticmethod
    def Zero(rows: Int, cols: Int) -> MatrixXd:
        return MatrixXd(rows, cols, 0.0)

    @staticmethod
    def zero(rows: Int, cols: Int) -> MatrixXd:
        return MatrixXd(rows, cols, 0.0)

    @staticmethod
    def Constant(rows: Int, cols: Int, val: Float64) -> MatrixXd:
        return MatrixXd(rows, cols, val)

    @always_inline
    @staticmethod
    def dtype() -> String:
        return "MatrixXd"


@fieldwise_init
struct VectorXd(Defaultable, Movable, Typeable):
    var _size: Int
    var _data: List[Float64]

    @always_inline
    def __init__(out self):
        self._size = 0
        self._data = List[Float64]()

    @always_inline
    def __init__(out self, size: Int, val: Float64 = 0.0):
        self._size = size
        self._data = List[Float64](length=size, fill=val)

    @always_inline
    def size(self) -> Int:
        return self._size

    @always_inline
    def rows(self) -> Int:
        return self._size

    @always_inline
    def cols(self) -> Int:
        return 1

    @always_inline
    def __getitem__(self, i: Int) -> Float64:
        return self._data[i]

    @always_inline
    def __setitem__(mut self, i: Int, val: Float64):
        self._data[i] = val

    @staticmethod
    def Zero[rows: Int](rows_hint: IntLiteral) -> Matrix[DType.float64, rows, 1]:
        return Matrix[DType.float64, rows, 1]()

    @staticmethod
    def zero[rows: Int](rows_hint: IntLiteral) -> Matrix[DType.float64, rows, 1]:
        return Matrix[DType.float64, rows, 1]()

    @staticmethod
    def Constant[rows: Int](
        rows_hint: IntLiteral, val: Float64
    ) -> Matrix[DType.float64, rows, 1]:
        return Matrix[DType.float64, rows, 1](val)

    @staticmethod
    def Zero(size: Int) -> VectorXd:
        return VectorXd(size, 0.0)

    @staticmethod
    def zero(size: Int) -> VectorXd:
        return VectorXd(size, 0.0)

    @staticmethod
    def Constant(size: Int, val: Float64) -> VectorXd:
        return VectorXd(size, val)

    @always_inline
    @staticmethod
    def dtype() -> String:
        return "VectorXd"


# Bridges MojoBridge's Matrix[T,rows,cols] (row-major storage) to layout's
# LayoutTensor (used by TrajectoryStateSoA/EigenSoA, expects col-major
# storage for a given Layout) -- two independently-ported Eigen equivalents
# with different memory layouts, so this copies element-by-element into a
# caller-owned buffer rather than reinterpreting the pointer directly.
# `buf` must stay alive for as long as the returned LayoutTensor is used.
def to_layout_tensor[
    T: DType, rows: Int, cols: Int
](
    m: Matrix[T, rows, cols], mut buf: InlineArray[Scalar[T], rows * cols]
) -> LayoutTensor[
    mut=True, T, Layout.col_major(rows, cols), MutAnyOrigin
]:
    for c in range(cols):
        for r in range(rows):
            buf[c * rows + r] = m[r, c]
    return LayoutTensor[
        mut=True, T, Layout.col_major(rows, cols), MutAnyOrigin
    ](buf.unsafe_ptr())
