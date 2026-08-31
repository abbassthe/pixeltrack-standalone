from std.memory.memory import _malloc
from std.sys import size_of
@deprecated(
    "This structure does not function correctly except in certain linear code"
    " sequences. Please refrain from using Static"
)
struct Static[T: Movable, id: Int = 0]:
    """This highly unsafe module allows you to use static variables within Mojo.
    You must ALWAYS use the init() function to initialize the backing the first time.
    Usage:
    ```mojo
    Static[Int].init()
    Static[Int].get() = 5
    # now Static[Int] (or alternatively, Static[Int, 0] is 5 everywhere in your program
    Static[Int].init()
    Static[Int, 3].get() = 23
    # now Static[Int, 3] is 23 everywhere in your program
    ```
    Note: May break as implementation of Mojo alias changes.
    """

    @staticmethod
    @always_inline("nodebug")
    def __get_backing() -> UnsafePointer[UnsafePointer[Self.T]]:
        """This function holds the actual object referenced in this static variable.
        """
        comptime __storage = _malloc[UnsafePointer[T]](size_of[UnsafePointer[T]]())
        return __storage

    @staticmethod
    @always_inline("nodebug")
    @deprecated(
        "This structure does not function correctly except in certain linear"
        " code sequences. Please refrain from using Static"
    )
    def unsafe_ptr() -> UnsafePointer[Self.T]:
        """Returns an unsafe pointer to the static object."""
        return Self.__get_backing()[]

    @staticmethod
    @always_inline("nodebug")
    @deprecated(
        "This structure does not function correctly except in certain linear"
        " code sequences. Please refrain from using Static"
    )
    def get() -> ref [MutableOrigin.cast_from[StaticConstantOrigin]] Self.T:
        """Returns a mutable reference to the static object."""
        return Self.__get_backing()[][]

    @staticmethod
    @always_inline("nodebug")
    @deprecated(
        "This structure does not function correctly except in certain linear"
        " code sequences. Please refrain from using Static"
    )
    def init(var item: Self.T):
        """Initializes the static object with a value. This function must ALWAYS be called before attempting to use the static object.
        """
        Self.__get_backing().init_pointee_move(UnsafePointer[T].alloc(1))
        Self.__get_backing()[].init_pointee_move(item^)
