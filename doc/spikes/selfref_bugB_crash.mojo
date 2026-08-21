# Bug B repro: projecting through an associated type on a self-referential
# parameter (Assoc = Self), then USING the resulting value.
#
# Shapes mirror MojoCudaDev/CUDACore/OneToManyAssoc.mojo, stripped of
# FlexiStorage/atomics/prefix-scan (Bug A is independent and already fixed).
#
# Expected: compiler crash (infinite recursion), no object file, no assembly.


trait AssocLike:
    comptime index_type: Copyable & Movable & ImplicitlyCopyable & Defaultable


struct View[Assoc: AssocLike](Movable):
    var assoc: UnsafePointer[Self.Assoc, MutAnyOrigin]
    # THE TRIGGER: field typed by projecting through the associated type.
    # OneToManyAssoc.mojo ships this as UnsafePointer[NoneType] precisely to
    # avoid what happens below.
    var contentStorage: UnsafePointer[Self.Assoc.index_type, MutAnyOrigin]

    fn __init__(out self):
        self.assoc = UnsafePointer[Self.Assoc, MutAnyOrigin]()
        self.contentStorage = UnsafePointer[
            Self.Assoc.index_type, MutAnyOrigin
        ]()

    fn __moveinit__(out self, deinit take: Self):
        self.assoc = take.assoc
        self.contentStorage = take.contentStorage


fn consume[T: AnyType](p: UnsafePointer[T, MutAnyOrigin]):
    pass


struct Assoc(Movable, AssocLike):
    comptime index_type = UInt32
    comptime View = View[Self]  # self-referential: Assoc = Self

    var dummy: Int

    fn __init__(out self):
        self.dummy = 0

    fn __moveinit__(out self, deinit take: Self):
        self.dummy = take.dummy

    fn initStorage(mut self, mut view: Self.View):
        # Merely accessing/discarding the field is reportedly fine; passing it
        # as a call argument forces type-checking of the projected type, which
        # is what crashes.
        consume(view.contentStorage)


def main():
    var a = Assoc()
    var v = Assoc.View()
    # A real call site is required -- a standalone definition does not
    # monomorphize far enough to trigger it.
    a.initStorage(v)
