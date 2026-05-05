from Framework.EDGetToken import EDGetTokenT
from Framework.EDPutToken import EDPutTokenT
from Framework.ProductRegistry import ProductRegistry
from MojoBridge.DTypes import Typeable

alias StreamID = Int32


struct WrapperBase(Copyable, Defaultable, Movable, Typeable):
    var _ptr: UnsafePointer[NoneType]

    @always_inline
    fn __init__(out self):
        self._ptr = UnsafePointer[NoneType]()

    @always_inline
    fn __copyinit__(out self, other: Self):
        self._ptr = other._ptr

    @always_inline
    fn __moveinit__(out self, var other: Self):
        self._ptr = other._ptr

    @always_inline
    fn product(self) -> UnsafePointer[NoneType, mut=False]:
        return self._ptr

    @staticmethod
    @always_inline
    fn dtype() -> String:
        return "WrapperBase"


fn det_blank(mut wrapper: WrapperBase):
    pass


struct Wrapper[T: Typeable & Movable](Movable, Typeable):
    var _ptr: UnsafePointer[T]

    @always_inline
    fn __init__(out self, var obj: T):
        self._ptr = UnsafePointer[T].alloc(1)
        self._ptr.init_pointee_move(obj^)

    @always_inline
    fn delete(self):
        self._ptr.destroy_pointee()
        self._ptr.free()

    @always_inline
    fn __moveinit__(out self, var other: Self):
        self._ptr = other._ptr

    @always_inline
    fn product(self) -> UnsafePointer[T, mut=False]:
        return self._ptr

    @staticmethod
    @always_inline
    fn dtype() -> String:
        return "Wrapper[" + T.dtype() + "]"


struct Event(Defaultable, Movable, Typeable):
    var _streamId: StreamID
    var _eventId: Int32
    var _products: List[WrapperBase]
    var _dets: List[fn (mut WrapperBase)]

    @always_inline
    fn __init__(out self):
        self._streamId = 0
        self._eventId = 0
        self._products = []
        self._dets = []

    @always_inline
    fn __init__(
        out self,
        var streamId: Int32,
        var eventId: Int32,
        ref reg: ProductRegistry,
    ):
        self._streamId = streamId
        self._eventId = eventId
        self._products = List[WrapperBase](
            length=reg.__len__(), fill=WrapperBase()
        )
        self._dets = List[fn (mut WrapperBase)](
            length=reg.__len__(), fill=det_blank
        )

    fn __del__(var self):
        for i in range(self._products.__len__()):
            self._dets[i](self._products[i])

    @always_inline
    fn __moveinit__(out self, var other: Self):
        self._streamId = other._streamId
        self._eventId = other._eventId
        self._products = other._products^
        self._dets = other._dets^

    @always_inline
    fn streamID(self) -> StreamID:
        return self._streamId

    @always_inline
    fn eventID(self) -> Int32:
        return self._eventId

    fn get[
        T: Typeable & Movable
    ](self, ref token: EDGetTokenT[T]) -> ref [self._products] T:
        return rebind[Wrapper[T]](self._products[token.index()]).product()[]

    # General variadic emplace is still blocked in Mojo 25.5: variadic pack
    # elements cannot be moved into a generic constructor. Accept an already
    # constructed product here, and let callers such as ScopedContextProduce
    # build the product before storing it.
    fn emplace[
        T: Typeable & Movable
    ](mut self, ref token: EDPutTokenT[T], var prod: T):
        @always_inline
        fn det[T: Typeable & Movable](mut wrapper: WrapperBase):
            rebind[Wrapper[T]](wrapper).delete()

        self._products[token.index()] = rebind[WrapperBase](Wrapper[T](prod^))
        self._dets[token.index()] = det[T]

    fn put[
        T: Typeable & Movable
    ](mut self, ref token: EDPutTokenT[T], var prod: T):
        self.emplace[T](token, prod^)

    @staticmethod
    @always_inline
    fn dtype() -> String:
        return "Event"
