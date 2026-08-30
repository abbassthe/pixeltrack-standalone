from sys import alignof, is_gpu
from bit import pop_count
from math import Ceilable, CeilDivable, Floorable, Truncable, sqrt
from utils.numerics import max_finite as _max_finite
from utils.numerics import max_or_inf as _max_or_inf
from utils.numerics import min_finite as _min_finite
from utils.numerics import min_or_neg_inf as _min_or_neg_inf
from hashlib.hasher import Hasher
from layout import Layout, LayoutTensor

from MojoCudaDev.MojoBridge.DTypes import Double, Typeable
from MojoCudaDev.MojoBridge.Vector import Vector


@fieldwise_init
struct _MatIterator[
    mat_mutability: Bool, //,
    W: DType,
    rows: Int,
    colns: Int,
    mat_origin: Origin[mut=mat_mutability],
    forward: Bool = True,
    row_wise: Bool = True,
](Copyable, ImplicitlyCopyable, Iterator, Movable, Typeable):
    alias mat_type = Matrix[Self.W, Self.rows, Self.colns]
    alias T = Scalar[Self.W]
    alias Element = Self.T

    var index: Int
    var src: Pointer[Self.mat_type, Self.mat_origin]

    fn __next_ref__(mut self) -> Self.T:
        @parameter
        if Self.forward:
            self.index += 1
            return self.src[][self.index - 1, Self.row_wise]
        else:
            self.index -= 1
            return self.src[][self.index, Self.row_wise]

    @always_inline
    fn __next__(mut self) -> Self.T:
        return self.__next_ref__()

    @always_inline
    fn __has_next__(self) -> Bool:
        return self.__len__() > 0

    @always_inline
    fn __iter__(self) -> Self:
        return self

    fn __len__(self) -> Int:
        @parameter
        if Self.forward:
            return len(self.src[]) - self.index
        else:
            return self.index

    @always_inline
    @staticmethod
    fn dtype() -> String:
        return (
            "_MatIterator["
            + String(Self.mat_mutability)
            + ", "
            + Self.W.__repr__()
            + ", "
            + String(Self.rows)
            + ", "
            + String(Self.colns)
            + ", Origin["
            + String(Self.mat_mutability)
            + "], "
            + String(Self.row_wise)
            + "]"
        )


# Common interface satisfied by both Matrix (owns its storage) and Map (a
# strided view over external storage), so generic code can accept either.
trait MatrixLike:
    alias ElemType: DType
    alias Rows: Int

    fn __getitem__(self, i: Int, j: Int) -> Scalar[Self.ElemType]:
        ...

    fn __setitem__(mut self, i: Int, j: Int, val: Scalar[Self.ElemType]):
        ...

    fn __getitem__(self, i: Int) -> Scalar[Self.ElemType]:
        ...

    fn __setitem__(mut self, i: Int, val: Scalar[Self.ElemType]):
        ...

    fn num_rows(self) -> Int:
        ...

    fn block[
        br: Int, bc: Int
    ](self, row: Int, col: Int) -> Matrix[Self.ElemType, br, bc]:
        ...

    fn head[n: Int](self) -> Matrix[Self.ElemType, n, 1]:
        ...

    fn col(self, c: Int) -> Vector[Self.ElemType, Self.Rows]:
        ...

    @staticmethod
    fn ColsAtCompileTime() -> Int:
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
    ImplicitlyCopyable,
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
    alias ElemType = Self.T
    alias Rows = Self.rows
    alias _L = List[List[Scalar[Self.T]]]
    alias _LS = InlineArray[InlineArray[Scalar[Self.T], Self.colns], Self.rows]
    alias _R = Vector[Self.T, Self.colns]
    alias _D = Scalar[Self.T]
    alias _DC = InlineArray[Vector[Self.T, Self.colns], Self.rows]
    alias _DB = InlineArray[Vector[DType.bool, colns], rows]
    alias _Mask = Matrix[DType.bool, rows, colns]
    var _data: Self._DC

    # SIMD specifics

    alias device_type: AnyType = Self

    fn _to_device_type[o: MutOrigin, //](self, target: UnsafePointer[NoneType, o]):
        target.bitcast[Self.device_type]()[] = self

    @staticmethod
    fn get_type_name() -> String:
        return (
            "Matrix[" + repr(T) + ", " + repr(rows) + ", " + repr(colns) + "]"
        )

    @staticmethod
    fn get_device_type_name() -> String:
        return Self.get_type_name()

    alias MAX = Self(_max_or_inf[T]())
    alias MIN = Self(_min_or_neg_inf[T]())
    alias MAX_FINITE = Self(_max_finite[T]())
    alias MIN_FINITE = Self(_min_finite[T]())

    alias _default_alignment = alignof[Self._D]() if is_gpu() else 1

    @doc_private
    @always_inline("nodebug")
    @implicit
    fn __init__(out self, value: __mlir_type.index, /):
        # support MLIR assignment for compatibility purposes
        self._data = Self._DC(fill=Self._R(value))

    # Lifecycle methods
    @always_inline
    fn __init__(out self):
        """Default constructor."""
        self._data = Self._DC(fill=Self._R())

    @always_inline
    fn __init__(out self, *, uninitialized: Bool):
        """Default unsafe constructor."""
        self._data = Self._DC(uninitialized=uninitialized)

    @always_inline
    fn copy(self) -> Self:
        """Explicitly construct a copy of self."""
        return self

    @always_inline
    fn __init__[U: DType, //](out self, *, var row: Vector[U, Self.colns]):
        """Initialize a matrix from a Vector row object of the same coln-size, splattered across all rows.
        """
        self._data = Self._DC(fill=Self._R(row))

    @always_inline
    fn __init__[U: DType, //](out self, *, var coln: Vector[U, Self.colns]):
        """Initialize a matrix from a Vector coln object of the same row-size, splattered across all columns.
        """
        self._data = Self._DC(uninitialized=True)

        @parameter
        for i in range(Self.rows):

            @parameter
            for j in range(Self.colns):
                self[i, j] = coln[i].cast[Self.T]()

    @always_inline
    fn __init__[U: DType, //](out self, val: Scalar[U], /):
        """Initializes a matrix with a scalar.
        The scalar is splatted across all the elements of the matrix."""
        self._data = Self._DC(fill=Self._R(val))

    @always_inline
    fn __init__(out self, val: Int, /):
        """Initializes a matrix with a signed integer.
        The signed integer is splatted across all the elements of the matrix."""
        self._data = Self._DC(fill=Self._R(val))

    @always_inline
    fn __init__(out self, val: UInt, /):
        """Initializes a matrix with a unsigned integer.
        The unsigned integer is splatted across all the elements of the matrix.
        """
        self._data = Self._DC(fill=Self._R(val))

    @always_inline
    @implicit
    fn __init__(out self, val: IntLiteral, /):
        """Initializes a matrix with an integer literal (implicit).
        The integer literal is splatted across all the elements of the matrix.
        """
        self._data = Self._DC(fill=Self._R(val))

    @always_inline
    @implicit
    fn __init__(out self, *values: Self._D, __list_literal__: () = ()):
        """Constructs a matrix via a variadic list of values in a literal format (implicit).
        """
        self._data = Self._DC(uninitialized=True)
        for i in range(values.__len__()):
            self[i] = values[i]

    @implicit
    fn __init__(out self, mat: Self._L):
        """Constructs a matrix via a matrix list representation (implicit)."""
        self._data = Self._DC(uninitialized=True)
        for i in range(min(Self.rows, mat.__len__())):
            for j in range(min(Self.colns, mat[0].__len__())):
                self[i, j] = mat[i][j]

    @implicit
    fn __init__(out self, mat: Self._LS):
        """Constructs a matrix via a matrix inline array representation (implicit).
        """
        self._data = Self._DC(uninitialized=True)

        @parameter
        for i in range(Self.rows):

            @parameter
            for j in range(Self.colns):
                self[i, j] = mat[i][j]

    @implicit
    fn __init__(out self, var data: Self._DC):
        """Constructs a matrix via a matrix inline array internal data object (implicit).
        """
        self._data = data^

    fn __init__[
        vrows: Int, vcolns: Int, //
    ](out self, mat: Matrix[Self.T, vrows, vcolns]):
        """Initialize a matrix from an arbitrary matrix. Might cause data loss.
        """
        self._data = Self._DC(uninitialized=True)

        @parameter
        for i in range(min(Self.rows * Self.colns, vrows * vcolns)):
            self[i] = mat[i]

    fn __init__[U: DType, //](out self, mat: Matrix[U, Self.rows, Self.colns]):
        """Initialize a matrix from a matrix of the same size of a different data type.
        """
        self._data = Self._DC(uninitialized=True)

        @parameter
        for i in range(Self.rows * Self.colns):
            self[i] = mat[i].cast[Self.T]()

    fn __init__[
        *, row_offset: Int, coln_offset: Int
    ](out self, mat: Matrix[Self.T, ...]):
        """Initializes a matrix as a slice of another matrix with specified output size and offset.
        """
        self._data = Self._DC(uninitialized=True)
        var u = 0

        @parameter
        for i in range(row_offset, Self.rows + row_offset):
            var v = 0

            @parameter
            for j in range(coln_offset, Self.colns + coln_offset):
                self[u, v] = mat[i, j]
                v += 1
            u += 1

    # Compatibility with V1 Matrices

    fn __init__[vsize: Int, //](out self, vec: Vector[Self.T, vsize]):
        """Initialize a matrix from an arbitrary vector (V1 format). Might cause data loss.
        """
        self._data = Self._DC(uninitialized=True)

        @parameter
        for i in range(min(Self.rows * Self.colns, vsize)):
            self[i] = vec[i]

    @implicit
    fn __init__(out self, values: List[Self._D], /):
        """Initialize a matrix from a list of values. Might cause data loss."""
        self._data = Self._DC(uninitialized=True)
        for i in range(min(self.__len__(), values.__len__())):
            self[i] = values[i]

    @always_inline
    fn __getitem__(self, i: Int) -> Self._D:
        return self._data[i // Self.colns][i % Self.colns]

    @always_inline
    fn __getitem__(self, i: Int, row_wise: Bool) -> Self._D:
        if row_wise:
            return self._data[i // Self.colns][i % Self.colns]
        else:
            return self._data[i % Self.rows][i // Self.rows]

    @always_inline
    fn __setitem__(mut self, i: Int, val: Self._D):
        self._data[i // Self.colns][i % Self.colns] = val

    @always_inline
    fn __setitem__(mut self, i: Int, row_wise: Bool, val: Self._D):
        if row_wise:
            self._data[i // Self.colns][i % Self.colns] = val
        else:
            self._data[i % Self.rows][i // Self.rows] = val

    @always_inline
    fn __len__(self) -> Int:
        return Self.rows * Self.colns

    @always_inline
    @staticmethod
    fn Zero() -> Self:
        return Self()

    @always_inline
    @staticmethod
    fn Constant(val: Self._D) -> Self:
        var res = Self()
        @parameter
        for i in range(Self.rows * Self.colns):
            res[i] = val
        return res

    @staticmethod
    fn RowsAtCompileTime() -> Int:
        return rows

    @staticmethod
    fn ColsAtCompileTime() -> Int:
        return Self.colns

    @always_inline
    fn num_rows(self) -> Int:
        return Self.rows

    @always_inline
    fn cols(self) -> Int:
        return Self.colns

    # Operators

    @always_inline
    fn __getitem__(self, i: Int, j: Int) -> Self._D:
        return self._data[i][j]

    @always_inline
    fn __setitem__(mut self, i: Int, j: Int, val: Self._D):
        self._data[i][j] = val

    fn __iter__(ref self) -> _MatIterator[T, rows, colns, origin_of(self)]:
        return _MatIterator[T, rows, colns, origin_of(self)](
            0, Pointer(to=self)
        )

    @always_inline
    fn __contains__(self, value: Self._D) -> Bool:
        var res = False

        @parameter
        for i in range(rows):
            res = res and self._data[i].__contains__(value)
            if res:
                return res
        return res

    @always_inline
    fn __add__(self, rhs: Self) -> Self:
        constrained[Self.T.is_numeric(), "DType must be numeric"]()
        var res = self

        @parameter
        for i in range(Self.rows):
            res._data[i] = res._data[i] + rhs._data[i]
        return res

    @always_inline
    fn __sub__(self, rhs: Self) -> Self:
        constrained[Self.T.is_numeric(), "DType must be numeric"]()
        var res = self

        @parameter
        for i in range(Self.rows):
            res._data[i] = res._data[i] - rhs._data[i]
        return res

    @always_inline
    fn __mul__[trp: Int, //](self, rhs: Matrix[Self.T, Self.colns, trp]) -> Matrix[Self.T, Self.rows, trp]:
        """Matrix product (rows x colns) @ (colns x trp), matching Eigen's operator*."""
        return self @ rhs

    @always_inline
    fn __mul__(self, scalar: Self._D) -> Self:
        constrained[Self.T.is_numeric(), "DType must be numeric"]()
        var res = self
        # Splat explicitly: relying on implicit Scalar->Vector conversion in
        # `res._data[i] * scalar` silently only fills lane 0 in this Mojo build.
        var splatted = Self._R(scalar)

        @parameter
        for i in range(Self.rows):
            res._data[i] = res._data[i] * splatted
        return res

    @always_inline
    fn __rmul__(self, scalar: Self._D) -> Self:
        return self * scalar

    @no_inline
    fn __matmul__[
        trp: Int, //
    ](self, rhs: Matrix[Self.T, Self.colns, trp]) -> Matrix[Self.T, Self.rows, trp]:
        var res = Matrix[Self.T, Self.rows, trp]()

        @parameter
        for i in range(Self.rows):

            @parameter
            for j in range(trp):
                res[i, j] = self._row_by_coln(rhs, i, j)
        return res

    @always_inline
    fn __truediv__(self, rhs: Self) -> Self:
        constrained[Self.T.is_numeric(), "DType must be numeric"]()
        var res = self

        @parameter
        for i in range(Self.rows):
            res._data[i] = res._data[i] / rhs._data[i]
        return res

    @always_inline
    fn __truediv__(self, scalar: Self._D) -> Self:
        constrained[Self.T.is_numeric(), "DType must be numeric"]()
        var res = self
        var splatted = Self._R(scalar)

        @parameter
        for i in range(Self.rows):
            res._data[i] = res._data[i] / splatted
        return res

    @always_inline
    fn __floordiv__(self, rhs: Self) -> Self:
        constrained[T.is_numeric(), "DType must be numeric"]()
        var res = self

        @parameter
        for i in range(rows):
            res._data[i] = res._data[i] // rhs._data[i]
        return res

    @always_inline
    fn __mod__(self, rhs: Self) -> Self:
        constrained[T.is_numeric(), "DType must be numeric"]()
        var res = self

        @parameter
        for i in range(rows):
            res._data[i] = res._data[i] % rhs._data[i]
        return res

    @no_inline
    fn __pow__(
        self: Matrix[T, rows, colns], exp: Int
    ) -> Matrix[T, rows, colns]:
        constrained[
            rows == colns, "Can only calculate power of a square matrix"
        ]()
        alias sq = Matrix[T, rows, rows]

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
    fn __lt__(self, rhs: Self) -> Self._Mask:
        var res = Self._Mask()

        @parameter
        for i in range(rows):
            res._data[i] = self._data[i] < rhs._data[i]
        return res

    @always_inline
    fn __le__(self, rhs: Self) -> Self._Mask:
        var res = Self._Mask()

        @parameter
        for i in range(rows):
            res._data[i] = self._data[i] <= rhs._data[i]
        return res

    @always_inline
    fn __eq__(self, rhs: Self) -> Self._Mask:
        var res = Self._Mask()

        @parameter
        for i in range(rows):
            res._data[i] = self._data[i] == rhs._data[i]
        return res

    @always_inline
    fn __ne__(self, rhs: Self) -> Self._Mask:
        var res = Self._Mask()

        @parameter
        for i in range(rows):
            res._data[i] = self._data[i] != rhs._data[i]
        return res

    @always_inline
    fn __gt__(self, rhs: Self) -> Self._Mask:
        var res = Self._Mask()

        @parameter
        for i in range(rows):
            res._data[i] = self._data[i] > rhs._data[i]
        return res

    @always_inline
    fn __ge__(self, rhs: Self) -> Self._Mask:
        var res = Self._Mask()

        @parameter
        for i in range(rows):
            res._data[i] = self._data[i] >= rhs._data[i]
        return res

    @always_inline
    fn __pos__(self) -> Self:
        constrained[T.is_numeric(), "DType must be numeric"]()
        return self

    @always_inline
    fn __neg__(self) -> Self:
        constrained[Self.T.is_numeric(), "DType must be numeric"]()
        var res = self

        @parameter
        for i in range(Self.rows):
            res[i] = -res[i]
        return res

    @always_inline
    fn __and__(self, rhs: Self) -> Self:
        constrained[
            T.is_integral() or T == DType.bool,
            "DType must be an integral or bool type",
        ]()
        var res = self

        @parameter
        for i in range(rows):
            res._data[i] = res._data[i] & rhs._data[i]
        return res

    @always_inline
    fn __xor__(self, rhs: Self) -> Self:
        constrained[
            T.is_integral() or T == DType.bool,
            "DType must be an integral or bool type",
        ]()
        var res = self

        @parameter
        for i in range(rows):
            res._data[i] = res._data[i] ^ rhs._data[i]
        return res

    @always_inline
    fn __or__(self, rhs: Self) -> Self:
        constrained[
            T.is_integral() or T == DType.bool,
            "DType must be an integral or bool type",
        ]()
        var res = self

        @parameter
        for i in range(rows):
            res._data[i] = res._data[i] | rhs._data[i]
        return res

    @always_inline
    fn __lshift__(self, rhs: Self) -> Self:
        constrained[T.is_integral(), "DType must be an integral type"]()
        var res = self

        @parameter
        for i in range(rows):
            res._data[i] = res._data[i] << rhs._data[i]
        return res

    @always_inline
    fn __rshift__(self, rhs: Self) -> Self:
        constrained[T.is_integral(), "DType must be an integral type"]()
        var res = self

        @parameter
        for i in range(rows):
            res._data[i] = res._data[i] >> rhs._data[i]
        return res

    @no_inline
    fn __invert__[
        W: DType, *, protect: Bool = False
    ](self: Matrix[Self.T, Self.rows, Self.colns]) -> Matrix[W, Self.rows, Self.colns]:
        constrained[Self.rows == Self.colns, "Can only find inverse of a square matrix"]()
        debug_assert(
            abs(self.det[DType.float64]()) > 1e-9, "Matrix is not invertible"
        )
        # if this assert fails, we'll return a weird value
        alias n = Self.rows

        var mat = self.cast[DType.float64]()
        var idn = Matrix[DType.float64, Self.rows, Self.colns].identity()

        @parameter
        for i in range(n):
            var pivot = i

            @parameter
            for j in range(i, n):
                if abs(mat[j, i]) > abs(mat[pivot, i]):
                    pivot = j

            mat._data[i], mat._data[pivot] = mat._data[pivot], mat._data[i]
            idn._data[i], idn._data[pivot] = idn._data[pivot], idn._data[i]

            if abs(mat[i, i]) < 1e-9:
                return idn.cast[W]()

            var div = mat[i, i]

            @parameter
            for j in range(n):
                mat[i, j] /= div
                idn[i, j] /= div

            @parameter
            for j in range(n):
                if i != j:
                    var mult = mat[j, i]

                    @parameter
                    for k in range(n):
                        mat[j, k] -= mult * mat[i, k]
                        idn[j, k] -= mult * idn[i, k]

        @parameter
        if protect:

            @parameter
            if W in (
                DType.uint8,
                DType.uint16,
                DType.uint32,
                DType.uint64,
                DType.uint128,
                DType.uint256,
            ):

                @parameter
                for i in range(n * n):
                    if idn[i] < 1e-9:
                        idn[i] = 0
        return idn.cast[W]()

    @always_inline
    fn __invert__(self) -> Self:
        return self.__invert__[Self.T]()

    # In place operations

    @always_inline("nodebug")
    fn __iadd__(mut self, rhs: Self):
        constrained[Self.T.is_numeric(), "DType must be numeric"]()
        self = self + rhs

    @always_inline("nodebug")
    fn __isub__(mut self, rhs: Self):
        constrained[T.is_numeric(), "DType must be numeric"]()
        self = self - rhs

    @always_inline("nodebug")
    fn __imul__(mut self, rhs: Self):
        constrained[Self.T.is_numeric(), "DType must be numeric"]()
        constrained[Self.rows == Self.colns, "In-place matrix multiply requires a square matrix"]()
        var result = rebind[Matrix[Self.T, Self.colns, Self.colns]](self) @ rebind[Matrix[Self.T, Self.colns, Self.colns]](rhs)
        self = rebind[Self](result)

    @always_inline("nodebug")
    fn __imul__(mut self, scalar: Self._D):
        constrained[Self.T.is_numeric(), "DType must be numeric"]()
        self = self * scalar

    @always_inline("nodebug")
    fn __itruediv__(mut self, rhs: Self):
        constrained[T.is_numeric(), "DType must be numeric"]()
        self = self / rhs

    @always_inline("nodebug")
    fn __itruediv__(mut self, scalar: Self._D):
        constrained[T.is_numeric(), "DType must be numeric"]()
        self = self / scalar

    @always_inline("nodebug")
    fn __ifloordiv__(mut self, rhs: Self):
        constrained[T.is_numeric(), "DType must be numeric"]()
        self = self // rhs

    @always_inline("nodebug")
    fn __imod__(mut self, rhs: Self):
        constrained[T.is_numeric(), "DType must be numeric"]()
        self = self.__mod__(rhs)

    @always_inline("nodebug")
    fn __ipow__(mut self: Matrix[T, rows, colns], rhs: Int):
        constrained[T.is_numeric(), "DType must be numeric"]()
        constrained[
            rows == colns, "Can only calculate power of a square matrix"
        ]()
        self = self.__pow__(rhs)

    @always_inline("nodebug")
    fn __iand__(mut self, rhs: Self):
        constrained[
            T.is_integral() or T == DType.bool,
            "DType must be an integral or bool type",
        ]()
        self = self & rhs

    @always_inline("nodebug")
    fn __ixor__(mut self, rhs: Self):
        constrained[
            T.is_integral() or T == DType.bool,
            "DType must be an integral or bool type",
        ]()
        self = self ^ rhs

    @always_inline("nodebug")
    fn __ior__(mut self, rhs: Self):
        constrained[
            T.is_integral() or T == DType.bool,
            "DType must be an integral or bool type",
        ]()
        self = self | rhs

    @always_inline("nodebug")
    fn __ilshift__(mut self, rhs: Self):
        constrained[T.is_integral(), "DType must be an integral type"]()
        self = self << rhs

    @always_inline("nodebug")
    fn __irshift__(mut self, rhs: Self):
        constrained[T.is_integral(), "DType must be an integral type"]()
        self = self >> rhs

    @always_inline("nodebug")
    fn __iinvert__(mut self):
        constrained[
            T.is_integral() or T == DType.bool,
            "DType must be an integral or bool type",
        ]()
        self = ~self

    # Reversed operations

    @always_inline
    fn __radd__(self, value: Self) -> Self:
        constrained[T.is_numeric(), "DType must be numeric"]()
        return value + self

    @always_inline
    fn __rsub__(self, value: Self) -> Self:
        constrained[T.is_numeric(), "DType must be numeric"]()
        return value - self

    @always_inline
    fn __rfloordiv__(self, rhs: Self) -> Self:
        constrained[T.is_numeric(), "DType must be numeric"]()
        return rhs // self

    @always_inline
    fn __rtruediv__(self, value: Self) -> Self:
        constrained[T.is_numeric(), "DType must be numeric"]()
        return value / self

    @always_inline
    fn __rmod__(self, value: Self) -> Self:
        constrained[T.is_numeric(), "DType must be numeric"]()
        return value % self

    @always_inline
    fn __rand__(self, value: Self) -> Self:
        constrained[
            T.is_integral() or T == DType.bool,
            "DType be an integral or bool type",
        ]()
        return value & self

    @always_inline
    fn __rxor__(self, value: Self) -> Self:
        constrained[
            T.is_integral() or T == DType.bool,
            "DType be an integral or bool type",
        ]()
        return value ^ self

    @always_inline
    fn __ror__(self, value: Self) -> Self:
        constrained[
            T.is_integral() or T == DType.bool,
            "DType be an integral or bool type",
        ]()
        return value | self

    @always_inline
    fn __rlshift__(self, value: Self) -> Self:
        constrained[T.is_integral(), "DType be an integral type"]()
        return value << self

    @always_inline
    fn __rrshift__(self, value: Self) -> Self:
        constrained[T.is_integral(), "DType be an integral type"]()
        return value >> self

    # Trait conformance

    @always_inline
    @staticmethod
    fn dtype() -> String:
        return (
            "Matrix["
            + Self.T.__repr__()
            + ", "
            + String(Self.rows)
            + ", "
            + String(Self.colns)
            + "]"
        )

    @always_inline
    fn __str__(self) -> String:
        return String.write(self)

    @no_inline
    fn __repr__(self) -> String:
        var output = String()
        output.write("Matrix[" + Self.T.__repr__() + ", ", Self.rows, ", ", Self.colns, "](")
        for i in range(self.__len__()):
            output.write(self[i])
            if i < self.__len__() - 1:
                output.write(", ")
        output.write(")")
        return output^

    @always_inline
    fn __floor__(self) -> Self:
        var res = Self()

        @parameter
        for i in range(Self.rows):
            res._data[i] = self._data[i].__floor__()
        return res

    @always_inline
    fn __ceil__(self) -> Self:
        var res = Self()

        @parameter
        for i in range(Self.rows):
            res._data[i] = self._data[i].__ceil__()
        return res

    @always_inline
    fn __trunc__(self) -> Self:
        var res = Self()

        @parameter
        for i in range(Self.rows):
            res._data[i] = self._data[i].__trunc__()
        return res

    @always_inline
    fn __abs__(self) -> Self:
        var res = Self()

        @parameter
        for i in range(Self.rows):
            res._data[i] = self._data[i].__abs__()
        return res

    @always_inline
    fn __round__(self) -> Self:
        var res = Self()

        @parameter
        for i in range(Self.rows):
            res._data[i] = self._data[i].__round__()
        return res

    @always_inline
    fn __round__(self, ndigits: Int) -> Self:
        var res = Self()

        @parameter
        for i in range(Self.rows):
            res._data[i] = self._data[i].__round__(ndigits)
        return res

    @always_inline
    fn __ceildiv__(self, denominator: Self) -> Self:
        return self.__truediv__(denominator).__round__()

    fn __hash__[H: Hasher](self, mut hasher: H):
        @parameter
        for i in range(Self.rows):
            self._data[i].__hash__[H](hasher)
        hasher._update_with_simd(Scalar[DType.uint64](37))

    # Methods

    @always_inline("nodebug")
    fn _refine[
        W: DType = Self.T, nrows: Int = Self.rows, ncolns: Int = Self.colns
    ](self) -> Matrix[W, nrows, ncolns]:
        return rebind[Matrix[W, nrows, ncolns]](self)

    @always_inline
    fn cast[target: DType](self) -> Matrix[target, Self.rows, Self.colns]:
        @parameter
        if Self.T == target:
            return self._refine[target]()

        @parameter
        if Self.T in (DType.float8_e4m3fn, DType.float8_e5m2):
            constrained[
                target
                in (
                    DType.bfloat16,
                    DType.float16,
                    DType.float32,
                    DType.float64,
                ),
                (
                    String(
                        (
                            "Only FP8->F64, FP8->F32, FP8->F16, and FP8->BF16"
                            " castings are implemented. "
                        ),
                        Self.T,
                        "->",
                        target,
                    )
                ),
            ]()

        # low level manip for efficiency
        var res = InlineArray[Vector[target, Self.colns], Self.rows](uninitialized=True)

        @parameter
        for i in range(Self.rows):
            res[i] = self._data[i].cast[target]()
        return res

    @always_inline
    fn is_power_of_two(self) -> Self._Mask:
        constrained[T.is_integral(), "DType must be integral"]()
        if T.is_unsigned():
            return self.pop_count() == 1
        else:
            return (self > 0) & (self & (self - 1) == 0)

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        writer.write("[")

        var width = 0

        @parameter
        for i in range(Self.rows * Self.colns):
            width = max(width, self[i].__str__().__len__())

        @parameter
        for i in range(Self.rows):
            if i != 0:
                writer.write(" ")
            writer.write("[")

            @parameter
            for j in range(Self.colns):
                var _c = width - self[i, j].__str__().__len__()
                writer.write(
                    " " * (_c if _c > 0 else 0)
                    + self[i, j].__str__()
                    + (" " if j < Self.colns - 1 else "")
                )
            writer.write("]")
            if i < Self.rows - 1:
                writer.write("\n")
        writer.write("]")

    fn row_iterator(
        ref self,
    ) -> _MatIterator[Self.T, Self.rows, Self.colns, origin_of(self)]:
        return _MatIterator[Self.T, Self.rows, Self.colns, origin_of(self)](
            0, Pointer(to=self)
        )

    fn coln_iterator(
        ref self,
    ) -> _MatIterator[T, rows, colns, origin_of(self), row_wise=False]:
        return _MatIterator[T, rows, colns, origin_of(self), row_wise=False](
            0, Pointer(to=self)
        )

    @always_inline
    fn row(self, i: Int) -> Vector[T, colns]:
        return self._data[i]

    @no_inline
    fn coln(self, j: Int) -> Vector[Self.T, Self.rows]:
        var res = Vector[Self.T, Self.rows]()

        @parameter
        for i in range(Self.rows):
            res[i] = self[i, j]
        return res

    @always_inline
    fn col(self, c: Int) -> Vector[Self.T, Self.rows]:
        return self.coln(c)

    @always_inline
    fn block[br: Int, bc: Int](self, row: Int, col: Int) -> Matrix[Self.T, br, bc]:
        var res = Matrix[Self.T, br, bc]()
        for r in range(br):
            for c in range(bc):
                res[r, c] = self[row + r, col + c]
        return res

    @always_inline
    fn set_block[
        br: Int, bc: Int
    ](mut self, row: Int, col: Int, val: Matrix[Self.T, br, bc]):
        """Writes val into self at (row, col), the write-back counterpart to
        block() (which returns a copy, unlike Eigen's reference-semantics
        .block())."""
        for r in range(br):
            for c in range(bc):
                self[row + r, col + c] = val[r, c]

    #this is not flattening it is just taking first col
    @always_inline
    fn head[n: Int](self) -> Matrix[Self.T, n, 1]:
        constrained[Self.colns == 1, "head() requires a column vector"]()
        var res = Matrix[Self.T, n, 1]()
        for i in range(n):
            res[i, 0] = self[i, 0]
        return res

    @always_inline
    fn squaredNorm(self) -> Self._D:
        var acc: Self._D = 0
        for i in range(Self.rows):
            for j in range(Self.colns):
                acc += self[i, j] * self[i, j]
        return acc

    @always_inline
    fn norm(self) -> Self._D:
        return sqrt(self.squaredNorm())

    @always_inline
    fn dot(self, other: Self) -> Self._D:
        var acc: Self._D = 0
        for i in range(Self.rows):
            for j in range(Self.colns):
                acc += self[i, j] * other[i, j]
        return acc

    @no_inline
    fn _row_by_coln(
        self, other: Matrix[Self.T, Self.colns, _], row: Int, coln: Int
    ) -> Self._D:
        constrained[Self.T.is_numeric(), "DType must be numeric"]()
        var sum: Self._D = 0

        @parameter
        for i in range(Self.colns):
            sum += self[row, i] * other[i, coln]
        return sum

    @no_inline
    fn transpose(self) -> Matrix[Self.T, Self.colns, Self.rows]:
        var res = Matrix[Self.T, Self.colns, Self.rows]()

        @parameter
        for i in range(Self.colns):

            @parameter
            for j in range(Self.rows):
                res[i, j] = self[j, i]
        return res

    @staticmethod
    @no_inline
    fn identity() -> Self:
        constrained[Self.rows == Self.colns, "Identity can only be a square matrix"]()
        var res = Self()
        for i in range(Self.rows):
            res[i, i] = 1
        return res

    @always_inline
    fn inverse[
        W: DType, *, protect: Bool = False
    ](self: Matrix[Self.T, Self.rows, Self.colns]) -> Matrix[W, Self.rows, Self.colns]:
        constrained[Self.rows == Self.colns, "Can only find inverse of a square matrix"]()
        return self.__invert__[W, protect=protect]()

    @always_inline
    fn inverse(self) -> Self:
        constrained[Self.rows == Self.colns, "Can only find inverse of a square matrix"]()
        return ~self

    @no_inline
    fn det[
        W: DType, *, protect: Bool = False
    ](self: Matrix[Self.T, Self.rows, Self.colns]) -> Scalar[W]:
        constrained[
            Self.rows == Self.colns, "Can only calculate determinant for a square matrix"
        ]()
        alias n = Self.rows

        var mat = self.cast[DType.float64]()
        var det: Double = 1.0

        @parameter
        for i in range(n):
            var pivot = i

            @parameter
            for j in range(i + 1, n):
                if abs(mat[j, i]) > abs(mat[pivot, i]):
                    pivot = j
            if pivot != i:
                mat._data[i], mat._data[pivot] = mat._data[pivot], mat._data[i]
                det *= -1
            if mat[i, i] == 0:
                return 0
            det *= mat[i, i]

            @parameter
            for j in range(i + 1, n):
                var factor: Double = mat[j, i] / mat[i, i]

                @parameter
                for k in range(i + 1, n):
                    mat[j, k] -= factor * mat[i, k]

        @parameter
        if protect:

            @parameter
            if W in (
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
    fn det[
        *, protect: Bool = False
    ](self: Matrix[Self.T, Self.rows, Self.colns],) -> Self._D:
        return self.det[Self.T, protect=protect]()

    @always_inline
    fn clamp(self, lower_bound: Self, upper_bound: Self) -> Self:
        var res = self

        @parameter
        for i in range(rows):
            res._data[i] = res._data[i].clamp(
                lower_bound._data[i], upper_bound._data[i]
            )
        return res

    @always_inline
    fn fma(self, multiplier: Self, accumulator: Self) -> Self:
        constrained[T.is_numeric(), "DType must be numeric"]()
        var res = self
        for i in range(rows):
            res._data[i] = res._data[i].fma(
                multiplier._data[i], accumulator._data[i]
            )
        return res

    fn slice[
        output_rows: Int,
        output_colns: Int,
        /,
        *,
        row_offset: Int = 0,
        coln_offset: Int = 0,
    ](self) -> Matrix[T, output_rows, output_colns]:
        constrained[
            0 <= row_offset < output_rows + row_offset <= rows,
            "Output rows must be a positive integer less than rows",
        ]()
        constrained[
            0 <= coln_offset < output_colns + coln_offset <= rows,
            "Output colns must be a positive integer less than colns",
        ]()

        @parameter
        if output_rows == 1 and output_colns == 1:
            return self[row_offset, coln_offset]

        return Matrix[T, output_rows, output_colns].__init__[
            row_offset=row_offset, coln_offset=coln_offset
        ](self)

    fn insert[
        *, row_offset: Int = 0, coln_offset: Int = 0
    ](self, mat: Matrix[Self.T, ...]) -> Self:
        alias input_rows = mat.rows
        alias input_colns = mat.colns
        constrained[
            0 <= row_offset < input_rows + row_offset <= rows,
            "Insertion position must not exceed the rows of the matrix",
        ]()
        constrained[
            0 <= coln_offset < input_colns + coln_offset <= rows,
            "Insertion position must not exceed the colns of the matrix",
        ]()

        @parameter
        if rows == 1 and colns == 1:
            constrained[
                input_rows == 1 and input_colns == 1,
                "The input width must be 1 if the size of the matrix is 1",
            ]()
            return mat[0]

        var res = self

        @parameter
        for i in range(row_offset, rows):
            res._data[i] = res._data[i].insert[offset=coln_offset](mat._data[i])
        return res

    fn iinsert[
        *, row_offset: Int = 0, coln_offset: Int = 0
    ](mut self, mat: Matrix[Self.T, ...]):
        alias input_rows = mat.rows
        alias input_colns = mat.colns
        constrained[
            0 <= row_offset < input_rows + row_offset <= rows,
            "Insertion position must not exceed the rows of the matrix",
        ]()
        constrained[
            0 <= coln_offset < input_colns + coln_offset <= rows,
            "Insertion position must not exceed the colns of the matrix",
        ]()

        @parameter
        if rows == 1 and colns == 1:
            constrained[
                input_rows == 1 and input_colns == 1,
                "The input width must be 1 if the size of the matrix is 1",
            ]()
            self[0] = mat[0]

        @parameter
        for i in range(row_offset, rows):
            self._data[i] = self._data[i].insert[offset=coln_offset](
                mat._data[i]
            )

    fn row_stack[
        mcolns: Int, //
    ](self, other: Matrix[T, rows, mcolns]) -> Matrix[T, rows, colns + mcolns]:
        var res = Matrix[T, rows, colns + mcolns](uninitialized=True)

        @parameter
        for i in range(rows):
            res._data[i] = self._data[i].join(other._data[i])
        return res

    fn coln_stack[
        mrows: Int, //
    ](self, other: Matrix[T, mrows, colns]) -> Matrix[T, rows + mrows, colns]:
        var res = Matrix[T, rows + mrows, colns](uninitialized=True)

        @parameter
        for i in range(rows):
            res._data[i] = self._data[i]

        @parameter
        for i in range(mrows):
            res._data[rows + i] = other._data[i]
        return res

    fn split[
        factor: Int = 2
    ](self) -> InlineArray[
        Matrix[T, rows // factor, colns // factor], factor * factor
    ]:
        constrained[
            rows == colns and rows % factor == 0,
            "Can only do integral splits on square matrices",
        ]()
        var res = InlineArray[
            Matrix[T, rows // factor, colns // factor], factor * factor
        ](uninitialized=True)
        var i = 0

        @parameter
        for row_offset in range(0, rows, rows // factor):

            @parameter
            for coln_offset in range(0, colns, colns // factor):
                res[i] = self.slice[
                    rows // factor,
                    colns // factor,
                    row_offset=row_offset,
                    coln_offset=coln_offset,
                ]()
                i += 1
        return res

    # Reductions

    fn reduce_max(self) -> Self._D:
        @parameter
        if Self.rows == 1 and Self.colns == 1:
            return self[0]
        var A = self._data[0].reduce_max()

        @parameter
        for i in range(Self.rows):
            A = max(A, self._data[i].reduce_max())
        return A

    fn reduce_min(self) -> Self._D:
        @parameter
        if Self.rows == 1 and Self.colns == 1:
            return self[0]
        var A = self._data[0].reduce_min()

        @parameter
        for i in range(Self.rows):
            A = min(A, self._data[i].reduce_min())
        return A

    fn reduce_add(self) -> Self._D:
        @parameter
        if Self.rows == 1 and Self.colns == 1:
            return self[0]
        var A = self._data[0].reduce_add()

        @parameter
        for i in range(1, Self.rows):
            A = A + self._data[i].reduce_add()
        return A

    fn reduce_mul(self) -> Self._D:
        @parameter
        if rows == 1 and colns == 1:
            return self[0]
        var A = self._data[0].reduce_mul()

        @parameter
        for i in range(1, rows):
            A = A * self._data[i].reduce_mul()
        return A

    fn reduce_and(self) -> Self._D:
        @parameter
        if rows == 1 and colns == 1:
            return self[0]
        var A = self._data[0].reduce_and()

        @parameter
        for i in range(rows):
            A = A & self._data[i].reduce_and()
        return A

    fn reduce_or(self) -> Self._D:
        @parameter
        if rows == 1 and colns == 1:
            return self[0]
        var A = self._data[0].reduce_or()

        @parameter
        for i in range(rows):
            A = A | self._data[i].reduce_or()
        return A

    fn reduce_bit_count(self) -> Int:
        constrained[
            T.is_integral() or T == DType.bool,
            "Expected either integral or bool type",
        ]()

        @parameter
        if T == DType.bool:
            return Int(self.cast[DType.uint8]().reduce_add())
        else:
            return Int(self.pop_count().reduce_add())

    fn pop_count(self) -> Self:
        var res = Self._DC(uninitialized=True)

        @parameter
        for i in range(rows):
            res[i] = Vector[T, colns](pop_count(self._data[i]._data))
        return Self(res)


struct Map[T: DType, rows: Int, colns: Int, default_inner_stride: Int = 1](
    Copyable,
    MatrixLike,
    Movable,
    Typeable,
):
    alias ElemType = Self.T
    alias Rows = Self.rows
    var data: UnsafePointer[Scalar[Self.T], MutAnyOrigin]
    var inner_stride: Int
    var outer_stride: Int

    @always_inline
    fn __init__(out self, ptr: UnsafePointer[Scalar[Self.T], MutAnyOrigin]):
        self.data = ptr
        self.inner_stride = Self.default_inner_stride
        self.outer_stride = Self.rows * Self.default_inner_stride

    @always_inline
    fn __init__(out self, ptr: UnsafePointer[Scalar[Self.T], MutAnyOrigin], inner_stride: Int):
        self.data = ptr
        self.inner_stride = inner_stride
        self.outer_stride = Self.rows * inner_stride

    @always_inline
    fn __init__(
        out self,
        ptr: UnsafePointer[Scalar[Self.T], MutAnyOrigin],
        inner_stride: Int,
        outer_stride: Int,
    ):
        self.data = ptr
        self.inner_stride = inner_stride
        self.outer_stride = outer_stride

    @always_inline
    fn __getitem__(self, r: Int, c: Int) -> Scalar[Self.T]:
        return (
            self.data + c * self.outer_stride + r * self.inner_stride
        )[]

    @always_inline
    fn __setitem__(mut self, r: Int, c: Int, val: Scalar[Self.T]):
        (self.data + c * self.outer_stride + r * self.inner_stride)[] = val
    #this is not flattening it is just taking first col
    @always_inline
    fn __getitem__(self, i: Int) -> Scalar[Self.T]:
        constrained[Self.colns == 1, "Map 1D access requires a column vector"]()
        return self[i, 0]

    @always_inline
    fn __setitem__(mut self, i: Int, val: Scalar[Self.T]):
        constrained[Self.colns == 1, "Map 1D access requires a column vector"]()
        self[i, 0] = val

    @always_inline
    fn col(self, c: Int) -> Vector[Self.T, Self.rows]:
        var res = Vector[Self.T, Self.rows]()
        for r in range(Self.rows):
            res[r] = self[r, c]
        return res

    @always_inline
    fn block[br: Int, bc: Int](self, row: Int, col: Int) -> Matrix[Self.T, br, bc]:
        var res = Matrix[Self.T, br, bc]()
        for r in range(br):
            for c in range(bc):
                res[r, c] = self[row + r, col + c]
        return res

    #this is not flattening it is just taking first col
    @always_inline
    fn head[n: Int](self) -> Matrix[Self.T, n, 1]:
        constrained[Self.colns == 1, "Map head() requires a column vector"]()
        var res = Matrix[Self.T, n, 1]()
        for i in range(n):
            res[i, 0] = self[i, 0]
        return res

    @staticmethod
    fn RowsAtCompileTime() -> Int:
        return rows

    @staticmethod
    fn ColsAtCompileTime() -> Int:
        return Self.colns

    @always_inline
    fn num_rows(self) -> Int:
        return Self.rows

    @always_inline
    @staticmethod
    fn dtype() -> String:
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
    fn __init__(out self):
        self._rows = 0
        self._cols = 0
        self._data = List[Float64]()

    @always_inline
    fn __init__(out self, rows: Int, cols: Int, val: Float64 = 0.0):
        self._rows = rows
        self._cols = cols
        self._data = List[Float64](length=rows * cols, fill=val)

    @always_inline
    fn rows(self) -> Int:
        return self._rows

    @always_inline
    fn cols(self) -> Int:
        return self._cols

    @always_inline
    fn __getitem__(self, r: Int, c: Int) -> Float64:
        return self._data[r + c * self._rows]

    @always_inline
    fn __setitem__(mut self, r: Int, c: Int, val: Float64):
        self._data[r + c * self._rows] = val

    @staticmethod
    fn Zero[rows: Int, cols: Int](
        rows_hint: IntLiteral, cols_hint: IntLiteral
    ) -> Matrix[DType.float64, rows, cols]:
        return Matrix[DType.float64, rows, cols]()

    @staticmethod
    fn zero[rows: Int, cols: Int](
        rows_hint: IntLiteral, cols_hint: IntLiteral
    ) -> Matrix[DType.float64, rows, cols]:
        return Matrix[DType.float64, rows, cols]()



    @staticmethod
    fn Zero(rows: Int, cols: Int) -> MatrixXd:
        return MatrixXd(rows, cols, 0.0)

    @staticmethod
    fn zero(rows: Int, cols: Int) -> MatrixXd:
        return MatrixXd(rows, cols, 0.0)

    @staticmethod
    fn Constant(rows: Int, cols: Int, val: Float64) -> MatrixXd:
        return MatrixXd(rows, cols, val)

    @always_inline
    @staticmethod
    fn dtype() -> String:
        return "MatrixXd"


@fieldwise_init
struct VectorXd(Defaultable, Movable, Typeable):
    var _size: Int
    var _data: List[Float64]

    @always_inline
    fn __init__(out self):
        self._size = 0
        self._data = List[Float64]()

    @always_inline
    fn __init__(out self, size: Int, val: Float64 = 0.0):
        self._size = size
        self._data = List[Float64](length=size, fill=val)

    @always_inline
    fn size(self) -> Int:
        return self._size

    @always_inline
    fn rows(self) -> Int:
        return self._size

    @always_inline
    fn cols(self) -> Int:
        return 1

    @always_inline
    fn __getitem__(self, i: Int) -> Float64:
        return self._data[i]

    @always_inline
    fn __setitem__(mut self, i: Int, val: Float64):
        self._data[i] = val

    @staticmethod
    fn Zero[rows: Int](rows_hint: IntLiteral) -> Matrix[DType.float64, rows, 1]:
        return Matrix[DType.float64, rows, 1]()

    @staticmethod
    fn zero[rows: Int](rows_hint: IntLiteral) -> Matrix[DType.float64, rows, 1]:
        return Matrix[DType.float64, rows, 1]()

    @staticmethod
    fn Constant[rows: Int](
        rows_hint: IntLiteral, val: Float64
    ) -> Matrix[DType.float64, rows, 1]:
        return Matrix[DType.float64, rows, 1](val)

    @staticmethod
    fn Zero(size: Int) -> VectorXd:
        return VectorXd(size, 0.0)

    @staticmethod
    fn zero(size: Int) -> VectorXd:
        return VectorXd(size, 0.0)

    @staticmethod
    fn Constant(size: Int, val: Float64) -> VectorXd:
        return VectorXd(size, val)

    @always_inline
    @staticmethod
    fn dtype() -> String:
        return "VectorXd"


# Bridges MojoBridge's Matrix[T,rows,cols] (row-major storage) to layout's
# LayoutTensor (used by TrajectoryStateSoA/EigenSoA, expects col-major
# storage for a given Layout) -- two independently-ported Eigen equivalents
# with different memory layouts, so this copies element-by-element into a
# caller-owned buffer rather than reinterpreting the pointer directly.
# `buf` must stay alive for as long as the returned LayoutTensor is used.
fn to_layout_tensor[
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
