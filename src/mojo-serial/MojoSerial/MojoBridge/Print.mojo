from std.collections import Set

from MojoSerial.MojoBridge.Matrix import Matrix


comptime StringStandardType = Copyable & Writable
comptime StringStandardKeyType = StringStandardType & KeyElement


def pprint[T1: StringStandardType, T2: StringStandardType, //](t: Tuple[T1, T2]):
    print("(" + String(t[0]) + ", " + String(t[1]) + ")")


def pprint[T: StringStandardType](L: List[T]):
    print("[", end="")
    if L.__len__() > 0:
        print(String(L[0]), end="")
        for i in range(1, len(L)):
            print(", " + String(L[i]), end="")
    print("]")


def pprint[T: StringStandardType, size: Int, //](L: InlineArray[T, size]):
    print("[", end="")

    if size > 0:
        print(String(L[0]), end="")

        for i in range(1, size):
            print(", " + String(L[i]), end="")
    print("]")


def pprint[T: StringStandardType, //](L: InlineArray[T, _], var ran: Int):
    print("[", end="")

    if ran > 0:
        print(String(L[0]), end="")

        for i in range(1, ran):
            print(", " + String(L[i]), end="")
    print("]")


def pprint[T1: StringStandardKeyType, T2: StringStandardType](D: Dict[T1, T2]):
    print("{", end="")
    if D.__len__() > 0:
        var skip = True
        for ref entry in D.items():
            if not skip:
                print(", ", end="")
            else:
                skip = False
            print(String(entry.key) + ": " + String(entry.value), end="")
    print("}")


def pprint[T: StringStandardKeyType](S: Set[T]):
    print("{", end="")
    if S.__len__() > 0:
        var skip = True
        for ref item in S:
            if not skip:
                print(", ", end="")
            else:
                skip = False
            print(String(item), end="")
    print("}")


def pprint[T: StringStandardType](o: Optional[T]):
    if o:
        print(String(o.value()))
    else:
        print("None")


def wprint[
    T: DType, //
](i: Scalar[T], *, width: Int = 0, end: StaticString = "\n"):
    var _w = String(i).byte_length()
    var _c = width - _w
    print(" " * (_c if _c > 0 else 0) + String(i), end=end)


def pprint[T: DType, //](M: Matrix[T, _, _]):
    var width: Int = 0
    for i in range(M.__len__()):
        width = max(width, String(M[i]).byte_length())
    for i in range(M.rows):
        print("[", end="")
        for j in range(M.colns):
            wprint(M[i, j], width=width, end=" ")
        print("\b]")

# `type()`/`tprint()` were dropped: they needed `reflection.get_type_name`,
# which 1.0 removed. `get_linkage_name` takes a function, not a type, so there
# is no replacement. Use the `Typeable.dtype()` trait for a type name instead.
