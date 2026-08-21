from memory import bitcast
from math import Ceilable, CeilDivable, Floorable, Truncable
from builtin.device_passable import DevicePassable
from sys import alignof, is_gpu
from std.sys.info import bit_width_of
from bit import pop_count
from utils.numerics import max_finite as _max_finite
from utils.numerics import max_or_inf as _max_or_inf
from utils.numerics import min_finite as _min_finite
from utils.numerics import min_or_neg_inf as _min_or_neg_inf
from hashlib.hasher import Hasher

from MojoCudaDev.MojoBridge.DTypes import Typeable


@always_inline
fn _pow_2[T: DType, //, n: Scalar[T]]() -> Scalar[T]:
    alias num_bits = bit_width_of[T]()
    var result = n - 1

    @parameter
    if num_bits > 1:
        result |= result >> 1

    @parameter
    if num_bits > 2:
        result |= result >> 2

    @parameter
    if num_bits > 4:
        result |= result >> 4

    @parameter
    if num_bits > 8:
        result |= result >> 8

    @parameter
    if num_bits > 16:
        result |= result >> 16

    @parameter
    if num_bits > 32:
        result |= result >> 32

    @parameter
    if num_bits > 64:
        result |= result >> 64
    return result + 1


@always_inline
fn _pow_2[n: Int]() -> Int:
    var result = n - 1
    result |= result >> 1
    result |= result >> 2
    result |= result >> 4
    result |= result >> 8
    result |= result >> 16
    result |= result >> 32
    return result + 1


@fieldwise_init
struct _VecIterator[
    vec_mutability: Bool, //,
    W: DType,
    size: Int,
    vec_origin: Origin[vec_mutability],
    forward: Bool = True,
](Copyable, Iterator, Movable, Typeable):
    alias vec_type = Vector[W, size]
    alias T = Scalar[W]
    alias Element = Self.T

    var index: Int
    var src: Pointer[Self.vec_type, vec_origin]

    fn __next_ref__(mut self) -> Self.T:
        @parameter
        if forward:
            self.index += 1
            return self.src[][self.index - 1]
        else:
            self.index -= 1
            return self.src[][self.index]

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
            "_VecIterator["
            + String(vec_mutability)
            + ", "
            + W.__repr__()
            + ", "
            + String(size)
            + ", Origin["
            + String(vec_mutability)
            + "], "
            + String(forward)
            + "]"
        )


@fieldwise_init
struct Vector[T: DType, size: Int](
    Absable,
    TrivialRegisterPassable,
    CeilDivable,
    Ceilable,
    Copyable,
    Defaultable,
    DevicePassable,
    Floorable,
    Hashable,
    Movable,
    Powable,
    Representable,
    Roundable,
    Sized,
    Stringable,
    Truncable,
    Typeable,
    Writable,
):
    alias psize = _pow_2[size]()
    alias _D = Scalar[T]
    alias _DC = SIMD[T, Self.psize]
    alias _Mask = Vector[DType.bool, size]
    var _data: Self._DC

    # SIMD specifics

    alias device_type: AnyTrivialRegType = Self

    fn _to_device_type(self, target: OpaquePointer):
        target.bitcast[Self.device_type]()[] = self

    @staticmethod
    fn get_type_name() -> String:
        return "Vector[" + repr(T) + ", " + repr(size) + "]"

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
        self._data = value

    # Lifecycle methods

    @always_inline
    fn __init__(out self):
        """Default constructor."""
        self._data = Self._DC()

    @always_inline
    fn copy(self) -> Self:
        """Explicitly construct a copy of self."""
        return Self.__copyinit__(self)

    @implicit
    fn __init__[vsize: Int, //](out self, vec: Vector[T, vsize]):
        """Initialize a vector from an arbitrary vector. Might cause data loss (implicit).
        """
        self = Self()

        @parameter
        for i in range(min(size, vsize)):
            self._data[i] = vec[i]

    @implicit
    fn __init__[vsize: Int, //](out self, vec: SIMD[T, vsize]):
        """Initialize a vector from an arbitrary SIMD vector. Might cause data loss (implicit).
        """
        self = Self()

        @parameter
        for i in range(min(size, vsize)):
            self._data[i] = vec[i]

    @always_inline
    @implicit
    fn __init__(out self, var vec: SIMD[T, size], /):
        """Initialize a vector from a SIMD object of the same size (implicit).
        """
        self._data = rebind[Self._DC](vec)

    @implicit
    fn __init__(out self, values: List[Self._D], /):
        """Initialize a vector from a list of values. Might cause data loss (implicit).
        """
        self = Self()
        for i in range(min(size, values.__len__())):
            self._data[i] = values[i]

    @always_inline
    @implicit
    fn __init__(out self, *values: Self._D, __list_literal__: () = ()):
        """Constructs a vector via a variadic list of values in a literal format.
        """
        self = Self()
        for i in range(values.__len__()):
            self._data[i] = values[i]

    fn __init__(out self, *values: Self._D):
        """Initialize a vector from a variadic list of values."""
        self = Self()
        for i in range(values.__len__()):
            self._data[i] = values[i]

    @always_inline
    fn __init__[U: DType, //](out self, val: Scalar[U], /):
        """Initializes a vector with a scalar.
        The scalar is splatted across all the elements of the vector."""
        self._data = Self._DC(val)

    @always_inline
    fn __init__(out self, val: Int, /):
        """Initializes a vector with a signed integer.
        The signed integer is splatted across all the elements of the vector."""
        self._data = Self._DC(val)

    @always_inline
    fn __init__(out self, val: UInt, /):
        """Initializes a vector with an unsigned integer.
        The unsigned integer is splatted across all the elements of the vector.
        """
        self._data = Self._DC(val)

    @always_inline
    @implicit
    fn __init__(out self, val: IntLiteral, /):
        """Initializes a vector with an integer literal (implicit).
        The signed integer is splatted across all the elements of the vector."""
        self._data = Self._DC(val)

    @always_inline
    fn __init__[U: DType, //](out self, value: SIMD[U, size], /):
        """Initializes a vector with a SIMD vector of the same size and of a different data type.
        """
        self._data = rebind[Self._DC](value.cast[T]())

    @always_inline
    fn __init__[U: DType, //](out self, vec: Vector[U, size], /):
        """Initializes a vector with a vector of the same size and of a different data type.
        """
        self._data = rebind[Self._DC](vec._data.cast[T]())

    fn __init__[*, offset: Int](out self, vec: Vector[T, _]):
        """Initializes a vector as a slice of another vector with specified output size and offset.
        """
        alias output_width = size

        self = Self()
        var i = 0

        @parameter
        for j in range(offset, offset + output_width):
            self._data[i] = vec._data[j]
            i += 1

    @staticmethod
    fn from_bits[U: DType, //](value: SIMD[U, size]) -> Vector[U, size]:
        """Initializes a vector from the bits of an integral SIMD vector."""
        constrained[U.is_integral(), "DType must be integral"]()
        return Vector[U, size](SIMD[U, size].from_bits(value))

    # Operators

    @always_inline
    fn __getitem__(self, idx: Int) -> Self._D:
        return self._data[idx]

    @always_inline
    fn __setitem__(mut self, idx: Int, val: Self._D):
        self._data[idx] = val

    fn __iter__(ref self) -> _VecIterator[T, size, __origin_of(self)]:
        return _VecIterator[T, size, __origin_of(self)](0, Pointer(to=self))

    @always_inline
    fn __contains__(self, value: Self._D) -> Bool:
        return self._data.__contains__(value)

    @always_inline
    fn __add__(self, rhs: Self) -> Self:
        constrained[T.is_numeric(), "DType must be numeric"]()
        return self._data + rhs._data

    @always_inline
    fn __sub__(self, rhs: Self) -> Self:
        constrained[T.is_numeric(), "DType must be numeric"]()
        return self._data - rhs._data

    @always_inline
    fn __mul__(self, rhs: Self) -> Self:
        constrained[T.is_numeric(), "DType must be numeric"]()
        return self._data * rhs._data

    @always_inline
    fn __matmul__(self, rhs: Self) -> Self._D:
        constrained[T.is_numeric(), "DType must be numeric"]()
        var res: Self._D = 0
        for i in range(size):
            res += self._data[i] * rhs._data[i]
        return res

    @always_inline
    fn __truediv__(self, rhs: Self) -> Self:
        constrained[T.is_numeric(), "DType must be numeric"]()
        return self._data / rhs._data

    @always_inline
    fn __floordiv__(self, rhs: Self) -> Self:
        constrained[T.is_numeric(), "DType must be numeric"]()
        return self._data // rhs._data

    @always_inline
    fn __mod__(self, rhs: Self) -> Self:
        constrained[T.is_numeric(), "DType must be numeric"]()
        return self._data % rhs._data

    @always_inline
    fn __pow__(self, exp: Int) -> Self:
        constrained[T.is_numeric(), "DType must be numeric"]()
        return self._data**exp

    @always_inline
    fn __pow__(self, exp: Self) -> Self:
        constrained[T.is_numeric(), "DType must be numeric"]()
        return self._data**exp._data

    @always_inline
    fn __lt__(self, rhs: Self) -> Self._Mask:
        return self._data < rhs._data

    @always_inline
    fn __le__(self, rhs: Self) -> Self._Mask:
        return self._data <= rhs._data

    @always_inline
    fn __eq__(self, rhs: Self) -> Self._Mask:
        return self._data == rhs._data

    @always_inline
    fn __ne__(self, rhs: Self) -> Self._Mask:
        return self._data != rhs._data

    @always_inline
    fn __gt__(self, rhs: Self) -> Self._Mask:
        return self._data > rhs._data

    @always_inline
    fn __ge__(self, rhs: Self) -> Self._Mask:
        return self._data >= rhs._data

    @always_inline
    fn __pos__(self) -> Self:
        constrained[T.is_numeric(), "DType must be numeric"]()
        return self

    @always_inline
    fn __neg__(self) -> Self:
        constrained[T.is_numeric(), "DType must be numeric"]()
        return -self._data

    @always_inline
    fn __and__(self, rhs: Self) -> Self:
        constrained[
            T.is_integral() or T is DType.bool,
            "DType must be an integral or bool type",
        ]()
        return self._data & rhs._data

    @always_inline
    fn __xor__(self, rhs: Self) -> Self:
        constrained[
            T.is_integral() or T is DType.bool,
            "DType must be an integral or bool type",
        ]()
        return self._data ^ rhs._data

    @always_inline
    fn __or__(self, rhs: Self) -> Self:
        constrained[
            T.is_integral() or T is DType.bool,
            "DType must be an integral or bool type",
        ]()
        return self._data | rhs._data

    @always_inline
    fn __lshift__(self, rhs: Self) -> Self:
        constrained[T.is_integral(), "DType must be an integral type"]()
        return self._data << rhs._data

    @always_inline
    fn __rshift__(self, rhs: Self) -> Self:
        constrained[T.is_integral(), "DType must be an integral type"]()
        return self._data >> rhs._data

    @always_inline
    fn __invert__(self) -> Self:
        constrained[
            T.is_integral() or T is DType.bool,
            "DType must be an integral or bool type",
        ]()
        return ~self._data

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
    fn __ipow__(mut self, rhs: Int):
        constrained[T.is_numeric(), "DType must be numeric"]()
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
    fn __rmatmul__(self, rhs: Self) -> Self._D:
        return rhs @ self

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
    fn __rpow__(self, base: Self) -> Self:
        constrained[T.is_numeric(), "DType must be numeric"]()
        return base**self

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
        return "Vector[" + T.__repr__() + ", " + String(size) + "]"

    @always_inline
    fn __len__(self) -> Int:
        return size

    @always_inline
    fn __str__(self) -> String:
        return String.write(self)

    @no_inline
    fn __repr__(self) -> String:
        var output = String()
        output.write("Vector[" + T.__repr__() + ", ", size, "](")
        for i in range(self.__len__()):
            output.write(self[i])
            if i < self.__len__() - 1:
                output.write(", ")
        output.write(")")
        return output^

    @always_inline
    fn __floor__(self) -> Self:
        return self._data.__floor__()

    @always_inline
    fn __ceil__(self) -> Self:
        return self._data.__ceil__()

    @always_inline
    fn __trunc__(self) -> Self:
        return self._data.__trunc__()

    @always_inline
    fn __abs__(self) -> Self:
        return self._data.__abs__()

    @always_inline
    fn __round__(self) -> Self:
        return self._data.__round__()

    @always_inline
    fn __round__(self, ndigits: Int) -> Self:
        return self._data.__round__(ndigits)

    @always_inline
    fn __ceildiv__(self, denominator: Self) -> Self:
        return self._data.__ceildiv__(denominator._data)

    fn __hash__[H: Hasher](self, mut hasher: H):
        hasher._update_with_simd(self._data)
        hasher._update_with_simd(Scalar[DType.uint64](37))

    # Methods

    @always_inline("nodebug")
    fn _refine[
        T: DType = Self.T, size: Int = Self.size
    ](self) -> Vector[T, size]:
        return rebind[Vector[T, size]](self)

    @always_inline
    fn cast[target: DType](self) -> Vector[target, size]:
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

        return self._data.cast[target]()

    @always_inline
    fn is_power_of_two(self) -> Self._Mask:
        constrained[T.is_integral(), "DType must be integral"]()

        @parameter
        if T.is_unsigned():
            return Self._Mask(pop_count(self._data) == 1)
        else:
            return (self > 0) & (self & (self - 1) == 0)

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        writer.write("[")
        for i in range(self.__len__()):
            writer.write(self[i])
            if i < self.__len__() - 1:
                writer.write(", ")
        writer.write("]")

    @always_inline
    fn clamp(self, lower_bound: Self, upper_bound: Self) -> Self:
        return self._data.clamp(lower_bound._data, upper_bound._data)

    @always_inline
    fn fma(self, multiplier: Self, accumulator: Self) -> Self:
        constrained[T.is_numeric(), "DType must be numeric"]()
        return self._data.fma(multiplier._data, accumulator._data)

    fn slice[
        output_width: Int, /, *, offset: Int = 0
    ](self) -> Vector[T, output_width]:
        constrained[
            0 <= offset < output_width + offset <= size,
            "Output width must be a positive integer less than size",
        ]()

        @parameter
        if output_width == 1:
            return self[offset]

        return Vector[T, output_width].__init__[offset=offset](self)

    fn insert[*, offset: Int = 0](self, value: Vector[T, _]) -> Self:
        alias input_width = value.size
        constrained[
            0 <= offset < input_width + offset <= size,
            "Insertion position must not exceed the size of the vector",
        ]()

        @parameter
        if size == 1:
            constrained[
                input_width == 1, "The input width must be 1 if the size is 1"
            ]()
            return value[0]

        return self._data.insert[offset=offset](value._data)

    fn iinsert[*, offset: Int = 0](mut self, value: Vector[T, _]):
        alias input_width = value.size
        constrained[
            0 <= offset < input_width + offset <= size,
            "Insertion position must not exceed the size of the vector",
        ]()

        @parameter
        if size == 1:
            constrained[
                input_width == 1, "The input width must be 1 if the size is 1"
            ]()
            self._data[0] = value[0]

        self._data = self._data.insert[offset=offset](value._data)

    fn join[
        vsize: Int, //
    ](self, other: Vector[T, vsize]) -> Vector[T, size + vsize]:
        var res = Vector[T, size + vsize]()
        res.iinsert(self)
        res.iinsert[offset=size](other)
        return res

    fn interleave[
        vsize: Int, //
    ](self, other: Vector[T, vsize]) -> Vector[T, size + vsize]:
        var res = Vector[T, size + vsize]()
        var u = 0
        var v = 0

        @parameter
        for i in range(min(size, vsize) * 2):
            if i % 2 == 0:
                res[i] = self[u]
                u += 1
            else:
                res[i] = other[v]
                v += 1

        @parameter
        if size > vsize:

            @parameter
            for i in range(vsize * 2, vsize + size):
                res[i] = self[u]
                u += 1

        @parameter
        if vsize > size:

            @parameter
            for i in range(size * 2, vsize + size):
                res[i] = other[v]
                v += 1
        return res

    @always_inline
    fn split(self) -> Tuple[Vector[T, size // 2], Vector[T, size // 2]]:
        constrained[
            size % 2 == 0 and size > 1,
            "Vector size must be divisible by 2 for splitting",
        ]()
        alias half_size = size // 2
        var se = self.slice[half_size]()
        var lf = self.slice[half_size, offset=half_size]()
        return se, lf

    @always_inline
    fn deinterleave(self) -> Tuple[Vector[T, size // 2], Vector[T, size // 2]]:
        constrained[
            size % 2 == 0 and size > 1,
            "Vector size must be divisible by 2 for deinterleaving",
        ]()

        @parameter
        if size == 2:
            return self[0], self[1]

        var res = Vector[T, size // 2](), Vector[T, size // 2]()

        @parameter
        for i in range(size // 2):
            res[0][i] = self[2 * i]
            res[1][i] = self[2 * i + 1]
        return res[0], res[1]

    fn reversed(self) -> Self:
        var res = self

        @parameter
        for i in range(size // 2):
            res[i], res[size - 1 - i] = res[size - 1 - i], res[i]
        return res

    fn pop_count(self) -> Self:
        return pop_count(self._data)

    # Reductions

    fn reduce_max(self) -> Self._D:
        @parameter
        if self.size == 1:
            return self._data[0]
        var A = self._data[0]

        @parameter
        for i in range(1, size):
            A = max(A, self._data[i])
        return A

    fn reduce_min(self) -> Self._D:
        @parameter
        if self.size == 1:
            return self._data[0]
        var A = self._data[0]

        @parameter
        for i in range(1, size):
            A = min(A, self._data[i])
        return A

    fn reduce_add(self) -> Self._D:
        @parameter
        if self.size == 1:
            return self._data[0]
        var A = self._data[0]

        @parameter
        for i in range(1, size):
            A = A + self._data[i]
        return A

    fn reduce_mul(self) -> Self._D:
        @parameter
        if self.size == 1:
            return self._data[0]
        var A = self._data[0]

        @parameter
        for i in range(1, size):
            A = A * self._data[i]
        return A

    fn reduce_and(self) -> Self._D:
        @parameter
        if self.size == 1:
            return self._data[0]
        var A = self._data[0]

        @parameter
        for i in range(1, size):
            A = A & self._data[i]
        return A

    fn reduce_or(self) -> Self._D:
        @parameter
        if self.size == 1:
            return self._data[0]
        var A = self._data[0]

        @parameter
        for i in range(1, size):
            A = A | self._data[i]
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
            return Int(Vector[T, size](pop_count(self._data)).reduce_add())
