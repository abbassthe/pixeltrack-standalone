from MojoSerial.Framework.EDGetToken import EDGetTokenT
from MojoSerial.Framework.EDPutToken import EDPutTokenT
from MojoSerial.Framework.ProductRegistry import ProductRegistry
from MojoSerial.MojoBridge.DTypes import Typeable

comptime StreamID = Int32


struct WrapperBase(Copyable, Defaultable, Movable, Typeable):
    var _ptr: UnsafePointer[NoneType]

    @always_inline
    def __init__(out self):
        self._ptr = UnsafePointer[NoneType]()

    @always_inline
    def __copyinit__(out self, other: Self):
        self._ptr = other._ptr

    @always_inline
    def __moveinit__(out self, var other: Self):
        self._ptr = other._ptr

    @always_inline
    def product(self) -> UnsafePointer[NoneType, mut=False]:
        return self._ptr

    @staticmethod
    @always_inline
    def dtype() -> String:
        return "WrapperBase"


def det_blank(mut wrapper: WrapperBase):
    pass


struct Wrapper[T: Typeable & Movable](Movable, Typeable):
    var _ptr: UnsafePointer[Self.T]

    @always_inline
    def __init__(out self, var obj: Self.T):
        self._ptr = UnsafePointer[T].alloc(1)
        self._ptr.init_pointee_move(obj^)

    @always_inline
    def delete(self):
        self._ptr.destroy_pointee()
        self._ptr.free()

    @always_inline
    def __moveinit__(out self, var other: Self):
        self._ptr = other._ptr

    @always_inline
    def product(self) -> UnsafePointer[Self.T, mut=False]:
        return self._ptr

    @staticmethod
    @always_inline
    def dtype() -> String:
        return "Wrapper[" + Self.T.dtype() + "]"


struct Event(Defaultable, Movable, Typeable):
    var _streamId: StreamID
    var _eventId: Int32
    var _products: List[WrapperBase]
    var _dets: List[def (mut WrapperBase)]

    @always_inline
    def __init__(out self):
        self._streamId = 0
        self._eventId = 0
        self._products = []
        self._dets = []

    @always_inline
    def __init__(
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
        self._dets = List[def (mut WrapperBase)](
            length=reg.__len__(), fill=det_blank
        )

    def __deinit__(var self):
        for i in range(self._products.__len__()):
            self._dets[i](self._products[i])

    @always_inline
    def __moveinit__(out self, var other: Self):
        self._streamId = other._streamId
        self._eventId = other._eventId
        self._products = other._products^
        self._dets = other._dets^

    @always_inline
    def streamID(self) -> StreamID:
        return self._streamId

    @always_inline
    def eventID(self) -> Int32:
        return self._eventId

    def get[
        T: Typeable & Movable
    ](self, ref token: EDGetTokenT[T]) -> ref [self._products] T:
        return rebind[Wrapper[T]](self._products[token.index()]).product()[]

    # emplace is not possible due to failure in binding the constructor at compile time, so we provide put instead

    def put[
        T: Typeable & Movable
    ](mut self, ref token: EDPutTokenT[T], var prod: T):
        @always_inline
        def det[T: Typeable & Movable](mut wrapper: WrapperBase):
            rebind[Wrapper[T]](wrapper).delete()

        self._products[token.index()] = rebind[WrapperBase](Wrapper[T](prod^))
        self._dets[token.index()] = det[T]

    @staticmethod
    @always_inline
    def dtype() -> String:
        return "Event"
