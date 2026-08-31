from MojoSerial.MojoBridge.DTypes import Typeable


struct ESWrapperBase(Copyable, Defaultable, Movable, Typeable):
    var _ptr: UnsafePointer[NoneType]

    @always_inline
    def __init__(out self):
        self._ptr = UnsafePointer[NoneType]()

    @always_inline
    def product(self) -> UnsafePointer[NoneType, mut=False]:
        return self._ptr

    @staticmethod
    @always_inline
    def dtype() -> String:
        return "ESWrapperBase"


def det_blank(mut wrapper: ESWrapperBase):
    pass


struct ESWrapper[T: Typeable & Movable](Movable, Typeable):
    var _ptr: UnsafePointer[Self.T]

    @always_inline
    def __init__(out self, var obj: Self.T):
        self._ptr = UnsafePointer[T].alloc(1)
        self._ptr.init_pointee_move(obj^)

    @always_inline
    def __moveinit__(out self, var other: Self):
        self._ptr = other._ptr

    @always_inline
    def delete(self):
        self._ptr.destroy_pointee()
        self._ptr.free()

    @always_inline
    def product(self) -> UnsafePointer[Self.T, mut=False]:
        return self._ptr

    @staticmethod
    @always_inline
    def dtype() -> String:
        return "ESWrapper[" + Self.T.dtype() + "]"


struct EventSetup(Defaultable, Movable, Typeable):
    var _typeToProduct: Dict[String, ESWrapperBase]
    var _dets: Dict[String, def (mut ESWrapperBase)]

    @always_inline
    def __init__(out self):
        self._typeToProduct = Dict[String, ESWrapperBase]()
        self._dets = Dict[String, def (mut ESWrapperBase)]()

    @always_inline
    def __moveinit__(out self, var other: Self):
        self._typeToProduct = other._typeToProduct^
        self._dets = other._dets^

    def __deinit__(var self):
        for k in self._typeToProduct.keys():
            try:
                self._dets[k](self._typeToProduct[k])
            except e:
                print(
                    (
                        "Error in Framework/EventSetup.mojo: Failed to delete"
                        " object with key"
                    ),
                    k,
                    "in EventSetup with error",
                    e,
                )

    def put[T: Typeable & Movable](mut self, var prod: T) raises:
        @always_inline
        def det[T: Typeable & Movable](mut wrapper: ESWrapperBase):
            rebind[ESWrapper[T]](wrapper).delete()

        if T.dtype() in self._typeToProduct:
            raise "RuntimeError: Product of type " + T.dtype() + " already exists."
        self._typeToProduct[T.dtype()] = rebind[ESWrapperBase](
            ESWrapper[T](prod^)
        )
        self._dets[T.dtype()] = det[T]

    def get[T: Typeable & Movable](self) raises -> ref [self._typeToProduct] T:
        if T.dtype() not in self._typeToProduct:
            raise "RuntimeError: Product of type " + T.dtype() + " is not produced."
        return rebind[ESWrapper[T]](self._typeToProduct[T.dtype()]).product()[]

    @staticmethod
    @always_inline
    def dtype() -> String:
        return "EventSetup"
