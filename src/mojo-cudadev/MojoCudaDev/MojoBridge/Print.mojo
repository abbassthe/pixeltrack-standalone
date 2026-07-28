from collections import Set
from compile.reflection import get_type_name

from MojoCudaDev.MojoBridge.Matrix import Matrix


alias StringStandardType = Copyable & Movable & Stringable
alias StringStandardKeyType = StringStandardType & KeyElement


fn pprint[T1: StringStandardType, T2: StringStandardType, //](t: Tuple[T1, T2]):
    print("(" + t[0].__str__() + ", " + t[1].__str__() + ")")


fn pprint[T: StringStandardType](L: List[T]):
    print("[", end="")
    if L.__len__() > 0:
        print(L[0].__str__(), end="")
        for i in range(1, len(L)):
            print(", " + L[i].__str__(), end="")
    print("]")


fn pprint[T: StringStandardType, size: Int, //](L: InlineArray[T, size]):
    print("[", end="")

    if size > 0:
        print(L[0].__str__(), end="")

        for i in range(1, size):
            print(", " + L[i].__str__(), end="")
    print("]")


fn pprint[T: StringStandardType, //](L: InlineArray[T, _], var ran: Int):
    print("[", end="")

    if ran > 0:
        print(L[0].__str__(), end="")

        for i in range(1, ran):
            print(", " + L[i].__str__(), end="")
    print("]")


fn pprint[T1: StringStandardKeyType, T2: StringStandardType](D: Dict[T1, T2]):
    print("{", end="")
    if D.__len__() > 0:
        var skip = True
        for ref entry in D.items():
            if not skip:
                print(", ", end="")
            else:
                skip = False
            print(entry.key.__str__() + ": " + entry.value.__str__(), end="")
    print("}")


fn pprint[T: StringStandardKeyType](S: Set[T]):
    print("{", end="")
    if S.__len__() > 0:
        var skip = True
        for ref item in S:
            if not skip:
                print(", ", end="")
            else:
                skip = False
            print(item.__str__(), end="")
    print("}")


fn pprint[T: StringStandardType](o: Optional[T]):
    if o:
        print(o.value().__str__())
    else:
        print("None")


fn wprint[
    T: DType, //
](i: Scalar[T], *, width: Int = 0, end: StaticString = "\n"):
    var _w = i.__str__().__len__()
    var _c = width - _w
    print(" " * (_c if _c > 0 else 0) + i.__str__(), end=end)


fn pprint[T: DType, //](M: Matrix[T, _, _]):
    var width: Int = 0
    for i in range(M.__len__()):
        width = max(width, M[i].__str__().__len__())
    for i in range(M.rows):
        print("[", end="")
        for j in range(M.colns):
            wprint(M[i, j], width=width, end=" ")
        print("\b]")


@always_inline
fn type[T: UnknownDestructibility](it: T, out type: String):
    return get_type_name[__type_of(it)]().split(".")[-1]


@always_inline
fn tprint[T: UnknownDestructibility](it: T):
    print(type(it))
