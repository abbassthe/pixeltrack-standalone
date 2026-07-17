from sys import alignof, is_gpu
from bit import pop_count
from math import Ceilable, CeilDivable, Floorable, Truncable
from utils.numerics import max_finite as _max_finite
from utils.numerics import max_or_inf as _max_or_inf
from utils.numerics import min_finite as _min_finite
from utils.numerics import min_or_neg_inf as _min_or_neg_inf
from hashlib.hasher import Hasher

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
    alias mat_type = Matrix[W, rows, colns]
    alias T = Scalar[W]
    alias Element = Self.T

    var index: Int
    var src: Pointer[Self.mat_type, mat_origin]

    fn __next_ref__(mut self) -> Self.T:
        @parameter
        if forward:
            self.index += 1
            return self.src[][self.index - 1, row_wise]
        else:
            self.index -= 1
            return self.src[][self.index, row_wise]

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
        if forward:
            return len(self.src[]) - self.index
        else:
            return self.index

    @always_inline
    @staticmethod
    fn dtype() -> String:
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


# A comment about this implementation: it is probably the speediest, but arguably not of the best memory efficiency (?)
# Handling rows in a SIMD structure does give rows immense advantage over columns, it also simplfies implementation... but we are still using InlineArray for memory
# TODO: Is implementing a matrix as an inline array of vectors faster or slower than a direct memory implementation using an unsafe pointer?
struct Matrix[T: DType, rows: Int, colns: Int](
    Absable,
    CeilDivable,
    Ceilable,
    Copyable,
    Defaultable,
    ExplicitlyCopyable,
    Floorable,
    Hashable,
    Movable,
    Representable,
    Roundable,
    Sized,
    Stringable,
    Truncable,
    Typeable,
    Writable,
):
    alias _L = List[List[Scalar[T]]]
    alias _LS = InlineArray[InlineArray[Scalar[T], colns], rows]
    alias _R = Vector[T, colns]
    alias _D = Scalar[T]
    alias _DC = InlineArray[Vector[T, colns], rows]
    alias _DB = InlineArray[Vector[DType.bool, colns], rows]
    alias _Mask = Matrix[DType.bool, rows, colns]
    var _data: Self._DC

    # SIMD specifics

    alias device_type: AnyType = Self

    fn _to_device_type(self, target: OpaquePointer):
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
        self._data = Self._DC(value)

    # Lifecycle methods
    @always_inline
    fn __init__(out self):
        """Default constructor."""
        self._data = Self._DC(Self._R())

    @always_inline
    fn __init__(out self, *, uninitialized: Bool):
        """Default unsafe constructor."""
        self._data = Self._DC(uninitialized=uninitialized)

    @always_inline
    fn copy(self) -> Self:
        """Explicitly construct a copy of self."""
        return Self.__copyinit__(self)

    @always_inline
    fn __init__[U: DType, //](out self, *, var row: Vector[U, colns]):
        """Initialize a matrix from a Vector row object of the same coln-size, splattered across all rows.
        """
        self._data = Self._DC(Self._R(row))

    @always_inline
    fn __init__[U: DType, //](out self, *, var coln: Vector[U, colns]):
        """Initialize a matrix from a Vector coln object of the same row-size, splattered across all columns.
        """
        self._data = Self._DC(uninitialized=True)

        @parameter
        for i in range(rows):

            @parameter
            for j in range(colns):
                self[i, j] = coln[i].cast[T]()

    @always_inline
    fn __init__[U: DType, //](out self, val: Scalar[U], /):
        """Initializes a matrix with a scalar.
        The scalar is splatted across all the elements of the matrix."""
        self._data = Self._DC(Self._R(val))

    @always_inline
    fn __init__(out self, val: Int, /):
        """Initializes a matrix with a signed integer.
        The signed integer is splatted across all the elements of the matrix."""
        self._data = Self._DC(Self._R(val))

    @always_inline
    fn __init__(out self, val: UInt, /):
        """Initializes a matrix with a unsigned integer.
        The unsigned integer is splatted across all the elements of the matrix.
        """
        self._data = Self._DC(Self._R(val))

    @always_inline
    @implicit
    fn __init__(out self, val: IntLiteral, /):
        """Initializes a matrix with an integer literal (implicit).
        The integer literal is splatted across all the elements of the matrix.
        """
        self._data = Self._DC(Self._R(val))

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
        for i in range(min(rows, mat.__len__())):
            for j in range(min(colns, mat[0].__len__())):
                self[i, j] = mat[i][j]

    @implicit
    fn __init__(out self, mat: Self._LS):
        """Constructs a matrix via a matrix inline array representation (implicit).
        """
        self._data = Self._DC(uninitialized=True)

        @parameter
        for i in range(rows):

            @parameter
            for j in range(colns):
                self[i, j] = mat[i][j]

    @implicit
    fn __init__(out self, var data: Self._DC):
        """Constructs a matrix via a matrix inline array internal data object (implicit).
        """
        self._data = data^

    fn __init__[
        vrows: Int, vcolns: Int, //
    ](out self, mat: Matrix[T, vrows, vcolns]):
        """Initialize a matrix from an arbitrary matrix. Might cause data loss.
        """
        self._data = Self._DC(uninitialized=True)

        @parameter
        for i in range(min(rows * colns, vrows * vcolns)):
            self[i] = mat[i]

    fn __init__[U: DType, //](out self, mat: Matrix[U, rows, colns]):
        """Initialize a matrix from a matrix of the same size of a different data type.
        """
        self._data = Self._DC(uninitialized=True)

        @parameter
        for i in range(rows * colns):
            self[i] = mat[i].cast[T]()

    fn __init__[
        *, row_offset: Int, coln_offset: Int
    ](out self, mat: Matrix[T, *_]):
        """Initializes a matrix as a slice of another matrix with specified output size and offset.
        """
        self._data = Self._DC(uninitialized=True)
        var u = 0

        @parameter
        for i in range(row_offset, rows + row_offset):
            var v = 0

            @parameter
            for j in range(coln_offset, colns + coln_offset):
                self[u, v] = mat[i, j]
                v += 1
            u += 1

    # Compatibility with V1 Matrices

    fn __init__[vsize: Int, //](out self, vec: Vector[T, vsize]):
        """Initialize a matrix from an arbitrary vector (V1 format). Might cause data loss.
        """
        self._data = Self._DC(uninitialized=True)

        @parameter
        for i in range(min(rows * colns, vsize)):
            self[i] = vec[i]

    @implicit
    fn __init__(out self, values: List[Self._D], /):
        """Initialize a matrix from a list of values. Might cause data loss."""
        self._data = Self._DC(uninitialized=True)
        for i in range(min(self.__len__(), values.__len__())):
            self[i] = values[i]

    @always_inline
    fn __getitem__(self, i: Int, row_wise: Bool = True) -> Self._D:
        if row_wise:
            return self._data[i // colns][i % colns]
        else:
            return self._data[i % rows][i // rows]

    @always_inline
    fn __setitem__(mut self, i: Int, val: Self._D):
        self._data[i // colns][i % colns] = val

    @always_inline
    fn __setitem__(mut self, i: Int, row_wise: Bool, val: Self._D):
        if row_wise:
            self._data[i // colns][i % colns] = val
        else:
            self._data[i % rows][i // rows] = val

    @always_inline
    fn __len__(self) -> Int:
        return rows * colns

    @always_inline
    @staticmethod
    fn Zero() -> Self:
        return Self()

    @always_inline
    @staticmethod
    fn Constant(val: Self._D) -> Self:
        var res = Self()
        @parameter
        for i in range(rows * colns):
            res[i] = val
        return res

    @staticmethod
    @parameter
    fn RowsAtCompileTime() -> Int:
        return rows

    @staticmethod
    @parameter
    fn ColsAtCompileTime() -> Int:
        return colns

    @always_inline
    fn num_rows(self) -> Int:
        return rows

    @always_inline
    fn cols(self) -> Int:
        return colns

    # Operators

    @always_inline
    fn __getitem__(self, i: Int, j: Int) -> Self._D:
        return self._data[i][j]

    @always_inline
    fn __setitem__(mut self, i: Int, j: Int, val: Self._D):
        self._data[i][j] = val

    fn __iter__(ref self) -> _MatIterator[T, rows, colns, __origin_of(self)]:
        return _MatIterator[T, rows, colns, __origin_of(self)](
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
        constrained[T.is_numeric(), "DType must be numeric"]()
        var res = self

        @parameter
        for i in range(rows):
            res._data[i] = res._data[i] + rhs._data[i]
        return res

    @always_inline
    fn __sub__(self, rhs: Self) -> Self:
        constrained[T.is_numeric(), "DType must be numeric"]()
        var res = self

        @parameter
        for i in range(rows):
            res._data[i] = res._data[i] - rhs._data[i]
        return res

    @always_inline
    fn __mul__(self, rhs: Self) -> Self:
        constrained[T.is_numeric(), "DType must be numeric"]()
        var res = self

        @parameter
        for i in range(rows):
            res._data[i] = res._data[i] * rhs._data[i]
        return res

    @no_inline
    fn __matmul__[
        trp: Int, //
    ](self, rhs: Matrix[T, colns, trp]) -> Matrix[T, rows, trp]:
        var res = Matrix[T, rows, trp]()

        @parameter
        for i in range(rows):

            @parameter
            for j in range(trp):
                res[i, j] = self._row_by_coln(rhs, i, j)
        return res

    @always_inline
    fn __truediv__(self, rhs: Self) -> Self:
        constrained[T.is_numeric(), "DType must be numeric"]()
        var res = self

        @parameter
        for i in range(rows):
            res._data[i] = res._data[i] / rhs._data[i]
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
        constrained[T.is_numeric(), "DType must be numeric"]()
        var res = self

        @parameter
        for i in range(rows):
            res[i] = -res[i]
        return res

    @always_inline
    fn __and__(self, rhs: Self) -> Self:
        constrained[
            T.is_integral() or T is DType.bool,
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
            T.is_integral() or T is DType.bool,
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
            T.is_integral() or T is DType.bool,
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
    ](self: Matrix[T, rows, colns]) -> Matrix[W, rows, colns]:
        constrained[rows == colns, "Can only find inverse of a square matrix"]()
        debug_assert(
            abs(self.det[DType.float64]()) > 1e-9, "Matrix is not invertible"
        )
        # if this assert fails, we'll return a weird value
        alias n = rows

        var mat = self.cast[DType.float64]()
        var idn = Matrix[DType.float64, rows, colns].identity()

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
        return self.__invert__[T]()

    # In place operations

    @always_inline("nodebug")
    fn __iadd__(mut self, rhs: Self):
        constrained[T.is_numeric(), "DType must be numeric"]()
        self = self + rhs

    @always_inline("nodebug")
    fn __isub__(mut self, rhs: Self):
        constrained[T.is_numeric(), "DType must be numeric"]()
        self = self - rhs

    @always_inline("nodebug")
    fn __imul__(mut self, rhs: Self):
        constrained[T.is_numeric(), "DType must be numeric"]()
        self = self * rhs

    @always_inline("nodebug")
    fn __itruediv__(mut self, rhs: Self):
        constrained[T.is_numeric(), "DType must be numeric"]()
        self = self / rhs

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
            T.is_integral() or T is DType.bool,
            "DType must be an integral or bool type",
        ]()
        self = self & rhs

    @always_inline("nodebug")
    fn __ixor__(mut self, rhs: Self):
        constrained[
            T.is_integral() or T is DType.bool,
            "DType must be an integral or bool type",
        ]()
        self = self ^ rhs

    @always_inline("nodebug")
    fn __ior__(mut self, rhs: Self):
        constrained[
            T.is_integral() or T is DType.bool,
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
            T.is_integral() or T is DType.bool,
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
    fn __rmul__(self, value: Self) -> Self:
        constrained[T.is_numeric(), "DType must be numeric"]()
        return value * self

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
            T.is_integral() or T is DType.bool,
            "DType be an integral or bool type",
        ]()
        return value & self

    @always_inline
    fn __rxor__(self, value: Self) -> Self:
        constrained[
            T.is_integral() or T is DType.bool,
            "DType be an integral or bool type",
        ]()
        return value ^ self

    @always_inline
    fn __ror__(self, value: Self) -> Self:
        constrained[
            T.is_integral() or T is DType.bool,
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
            + T.__repr__()
            + ", "
            + String(rows)
            + ", "
            + String(colns)
            + "]"
        )

    @always_inline
    fn __str__(self) -> String:
        return String.write(self)

    @no_inline
    fn __repr__(self) -> String:
        var output = String()
        output.write("Matrix[" + T.__repr__() + ", ", rows, ", ", colns, "](")
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
        for i in range(rows):
            res._data[i] = self._data[i].__floor__()
        return res

    @always_inline
    fn __ceil__(self) -> Self:
        var res = Self()

        @parameter
        for i in range(rows):
            res._data[i] = self._data[i].__ceil__()
        return res

    @always_inline
    fn __trunc__(self) -> Self:
        var res = Self()

        @parameter
        for i in range(rows):
            res._data[i] = self._data[i].__trunc__()
        return res

    @always_inline
    fn __abs__(self) -> Self:
        var res = Self()

        @parameter
        for i in range(rows):
            res._data[i] = self._data[i].__abs__()
        return res

    @always_inline
    fn __round__(self) -> Self:
        var res = Self()

        @parameter
        for i in range(rows):
            res._data[i] = self._data[i].__round__()
        return res

    @always_inline
    fn __round__(self, ndigits: Int) -> Self:
        var res = Self()

        @parameter
        for i in range(rows):
            res._data[i] = self._data[i].__round__(ndigits)
        return res

    @always_inline
    fn __ceildiv__(self, denominator: Self) -> Self:
        return self.__truediv__(denominator).__round__()

    fn __hash__[H: Hasher](self, mut hasher: H):
        @parameter
        for i in range(rows):
            self._data[i].__hash__[H](hasher)
        hasher._update_with_simd(Scalar[DType.uint64](37))

    # Methods

    @always_inline("nodebug")
    fn _refine[
        T: DType = Self.T, rows: Int = Self.rows, colns: Int = Self.colns
    ](self) -> Matrix[T, rows, colns]:
        return rebind[Matrix[T, rows, colns]](self)

    @always_inline
    fn cast[target: DType](self) -> Matrix[target, rows, colns]:
        @parameter
        if T is target:
            return self._refine[target]()

        @parameter
        if T in (DType.float8_e4m3fn, DType.float8_e5m2):
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
                        T,
                        "->",
                        target,
                    )
                ),
            ]()

        # low level manip for efficiency
        var res = InlineArray[Vector[target, colns], rows](uninitialized=True)

        @parameter
        for i in range(rows):
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
        for i in range(rows * colns):
            width = max(width, self[i].__str__().__len__())

        @parameter
        for i in range(rows):
            if i != 0:
                writer.write(" ")
            writer.write("[")

            @parameter
            for j in range(colns):
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

    fn row_iterator(
        ref self,
    ) -> _MatIterator[T, rows, colns, __origin_of(self)]:
        return _MatIterator[T, rows, colns, __origin_of(self)](
            0, Pointer(to=self)
        )

    fn coln_iterator(
        ref self,
    ) -> _MatIterator[T, rows, colns, __origin_of(self), row_wise=False]:
        return _MatIterator[T, rows, colns, __origin_of(self), row_wise=False](
            0, Pointer(to=self)
        )

    @always_inline
    fn row(self, i: Int) -> Vector[T, colns]:
        return self._data[i]

    @no_inline
    fn coln(self, j: Int) -> Vector[T, rows]:
        var res = Vector[T, rows]()

        @parameter
        for i in range(rows):
            res[i] = self[i, j]
        return res

    @no_inline
    fn _row_by_coln(
        self, other: Matrix[T, colns, _], row: Int, coln: Int
    ) -> Self._D:
        constrained[T.is_integral(), "DType must be an integral type"]()
        var sum: Self._D = 0

        @parameter
        for i in range(colns):
            sum += self[row, i] * other[i, coln]
        return sum

    @no_inline
    fn transpose(self) -> Matrix[T, colns, rows]:
        var res = Matrix[T, colns, rows]()

        @parameter
        for i in range(colns):

            @parameter
            for j in range(rows):
                res[i, j] = self[j, i]
        return res

    @staticmethod
    @no_inline
    fn identity() -> Self:
        constrained[rows == colns, "Identity can only be a square matrix"]()
        var res = Self()
        for i in range(rows):
            res[i, i] = 1
        return res

    @always_inline
    fn inverse[
        W: DType, *, protect: Bool = False
    ](self: Matrix[T, rows, colns]) -> Matrix[W, rows, colns]:
        constrained[rows == colns, "Can only find inverse of a square matrix"]()
        return self.__invert__[W, protect=protect]()

    @always_inline
    fn inverse(self) -> Self:
        constrained[rows == colns, "Can only find inverse of a square matrix"]()
        return ~self

    @no_inline
    fn det[
        W: DType, *, protect: Bool = False
    ](self: Matrix[T, rows, colns]) -> Scalar[W]:
        constrained[
            rows == colns, "Can only calculate determinant for a square matrix"
        ]()
        alias n = rows

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
    ](self: Matrix[T, rows, colns],) -> Self._D:
        return self.det[T, protect=protect]()

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
    ](self, mat: Matrix[T, *_]) -> Self:
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
    ](mut self, mat: Matrix[T, *_]):
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
        if rows == 1 and colns == 1:
            return self[0]
        var A = self._data[0].reduce_max()

        @parameter
        for i in range(rows):
            A = max(A, self._data[i].reduce_max())
        return A

    fn reduce_min(self) -> Self._D:
        @parameter
        if rows == 1 and colns == 1:
            return self[0]
        var A = self._data[0].reduce_min()

        @parameter
        for i in range(rows):
            A = min(A, self._data[i].reduce_min())
        return A

    fn reduce_add(self) -> Self._D:
        @parameter
        if rows == 1 and colns == 1:
            return self[0]
        var A = self._data[0].reduce_add()

        @parameter
        for i in range(rows):
            A = A + self._data[i].reduce_add()
        return A

    fn reduce_mul(self) -> Self._D:
        @parameter
        if rows == 1 and colns == 1:
            return self[0]
        var A = self._data[0].reduce_mul()

        @parameter
        for i in range(rows):
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
            T.is_integral() or T is DType.bool,
            "Expected either integral or bool type",
        ]()

        @parameter
        if T is DType.bool:
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
    Movable,
    Typeable,
):
    var data: UnsafePointer[Scalar[T]]
    var inner_stride: Int
    var outer_stride: Int

    @always_inline
    fn __init__(out self, ptr: UnsafePointer[Scalar[T]]):
        self.data = ptr
        self.inner_stride = default_inner_stride
        self.outer_stride = rows * default_inner_stride

    @always_inline
    fn __init__(out self, ptr: UnsafePointer[Scalar[T]], inner_stride: Int):
        self.data = ptr
        self.inner_stride = inner_stride
        self.outer_stride = rows * inner_stride

    @always_inline
    fn __init__(
        out self,
        ptr: UnsafePointer[Scalar[T]],
        inner_stride: Int,
        outer_stride: Int,
    ):
        self.data = ptr
        self.inner_stride = inner_stride
        self.outer_stride = outer_stride

    @always_inline
    fn __getitem__(self, r: Int, c: Int) -> Scalar[T]:
        return (
            self.data + c * self.outer_stride + r * self.inner_stride
        )[]

    @always_inline
    fn __setitem__(mut self, r: Int, c: Int, val: Scalar[T]):
        (self.data + c * self.outer_stride + r * self.inner_stride)[] = val
    #this is not flattening it is just taking first col
    @always_inline
    fn __getitem__(self, i: Int) -> Scalar[T]:
        constrained[colns == 1, "Map 1D access requires a column vector"]()
        return self[i, 0]

    @always_inline
    fn __setitem__(mut self, i: Int, val: Scalar[T]):
        constrained[colns == 1, "Map 1D access requires a column vector"]()
        self[i, 0] = val

    @always_inline
    fn col(self, c: Int) -> Vector[T, rows]:
        var res = Vector[T, rows]()
        for r in range(rows):
            res[r] = self[r, c]
        return res

    @always_inline
    fn block[br: Int, bc: Int](self, row: Int, col: Int) -> Matrix[T, br, bc]:
        var res = Matrix[T, br, bc]()
        for r in range(br):
            for c in range(bc):
                res[r, c] = self[row + r, col + c]
        return res

    #this is not flattening it is just taking first col
    @always_inline
    fn head[n: Int](self) -> Matrix[T, n, 1]:
        constrained[colns == 1, "Map head() requires a column vector"]()
        var res = Matrix[T, n, 1]()
        for i in range(n):
            res[i, 0] = self[i, 0]
        return res

    @staticmethod
    @parameter
    fn RowsAtCompileTime() -> Int:
        return rows

    @staticmethod
    @parameter
    fn ColsAtCompileTime() -> Int:
        return colns

    @always_inline
    @staticmethod
    fn dtype() -> String:
        return (
            "Map["
            + T.__repr__()
            + ", "
            + String(rows)
            + ", "
            + String(colns)
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
