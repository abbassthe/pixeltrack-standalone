from std.collections.dict import _DictKeyIter
from std.pathlib import Path

from MojoSerial.Framework.ESProducer import ESProducer
from MojoSerial.Framework.EventSetup import EventSetup
from MojoSerial.MojoBridge.DTypes import Typeable


@fieldwise_init
struct ESProducerWrapper(Copyable, Defaultable, Movable, Typeable):
    var _ptr: UnsafePointer[NoneType]

    @always_inline
    def __init__(out self):
        self._ptr = UnsafePointer[NoneType]()

    @always_inline
    def producer(self) -> UnsafePointer[NoneType]:
        return self._ptr

    @staticmethod
    @always_inline
    def dtype() -> String:
        return "ESProducerWrapper"


struct ESProducerWrapperT[T: Typeable & ESProducer](Movable, Typeable):
    var _ptr: UnsafePointer[Self.T]

    @always_inline
    def __init__(out self, var path: Path):
        self._ptr = UnsafePointer[T].alloc(1)
        __get_address_as_uninit_lvalue(self._ptr.address) = Self.T.__init__(path)

    @always_inline
    def __init__(out self, *, deinit move: Self):
        self._ptr = move._ptr

    @always_inline
    def delete(self):
        self._ptr.destroy_pointee()
        self._ptr.free()

    @always_inline
    def producer(self) -> UnsafePointer[Self.T]:
        return self._ptr

    @staticmethod
    @always_inline
    def dtype() -> String:
        return "ESProducerWrapperT[" + Self.T.dtype() + "]"


struct ESProducerConcrete(Copyable, Movable, Typeable):
    comptime _C = def (var Path) -> ESProducerWrapper
    comptime _P = def (mut ESProducerWrapper, mut EventSetup)
    comptime _D = def (mut ESProducerWrapper)
    var _producer: ESProducerWrapper
    var _create: Self._C
    var _produce: Self._P
    var _det: Self._D

    @always_inline
    def __init__(out self, create: Self._C, produce: Self._P, det: Self._D):
        self._producer = ESProducerWrapper()
        self._create = create
        self._produce = produce
        self._det = det

    @always_inline
    def delete(mut self):
        self._det(self._producer)

    @always_inline
    def create(mut self, var path: Path):
        self._producer = self._create(path^)

    @always_inline
    def produce(mut self, mut eventSetup: EventSetup):
        self._produce(self._producer, eventSetup)

    @staticmethod
    @always_inline
    def dtype() -> String:
        return "ESProducerConcrete"


struct Registry(Typeable):
    comptime _pluginRegistryType = Dict[String, ESProducerConcrete]
    var _pluginRegistry: Self._pluginRegistryType

    @always_inline
    def __init__(out self):
        self._pluginRegistry = {}

    @always_inline
    def __deinit__(var self):
        self.delete()

    @always_inline
    def __getitem__(self, var name: String) raises -> ESProducerConcrete:
        return self._pluginRegistry[name^]

    @always_inline
    def __setitem__(
        mut self, var name: String, var esproducer: ESProducerConcrete
    ) raises:
        self._pluginRegistry[name^] = esproducer^

    @always_inline
    def delete(mut self):
        for i in range(self._pluginRegistry._entries.__len__()):
            if self._pluginRegistry._entries[i]:
                self._pluginRegistry._entries[i].unsafe_value().value.delete()

    @staticmethod
    @always_inline
    def dtype() -> String:
        return "Registry"
struct ESPluginFactory:
    @staticmethod
    @always_inline
    def getAll(
        mut reg: Registry,
    ) -> _DictKeyIter[
        Registry._pluginRegistryType.K,
        Registry._pluginRegistryType.V,
        Registry._pluginRegistryType.H,
        origin_of(reg._pluginRegistry),
    ]:
        return reg._pluginRegistry.keys()

    @staticmethod
    @always_inline
    def size(mut reg: Registry) -> Int:
        return reg._pluginRegistry.__len__()

    @staticmethod
    @always_inline
    def create(
        var name: String, var path: Path, mut reg: Registry
    ) raises -> ESProducerConcrete:
        reg[name].create(path^)
        return reg[name^]


@always_inline
def fwkEventSetupModule[T: Typeable & ESProducer](mut reg: Registry):
    @always_inline
    def create_templ[
        T: Typeable & ESProducer
    ](var path: Path) -> ESProducerWrapper:
        return rebind[ESProducerWrapper](ESProducerWrapperT[T](path^))

    @always_inline
    def produce_templ[
        T: Typeable & ESProducer
    ](mut esproducer: ESProducerWrapper, mut eventSetup: EventSetup):
        rebind[ESProducerWrapperT[T]](esproducer).producer()[].produce(
            eventSetup
        )

    @always_inline
    def det_templ[T: Typeable & ESProducer](mut esproducer: ESProducerWrapper):
        rebind[ESProducerWrapperT[T]](esproducer).delete()

    var crp = ESProducerConcrete(
        create_templ[T], produce_templ[T], det_templ[T]
    )
    try:
        reg[T.dtype()] = crp^
    except e:
        print(
            "Framework/ESPluginFactory.mojo, failed to register plugin ",
            T.dtype(),
            ", got error: ",
            e,
            sep="",
        )
