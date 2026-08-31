from std.collections.dict import _DictKeyIter

from MojoSerial.Framework.EDProducer import EDProducer
from MojoSerial.Framework.Event import Event
from MojoSerial.Framework.EventSetup import EventSetup
from MojoSerial.Framework.ProductRegistry import ProductRegistry
from MojoSerial.MojoBridge.DTypes import Typeable


@fieldwise_init
struct EDProducerWrapper(Copyable, Defaultable, Movable, Typeable):
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
        return "EDProducerWrapper"


struct EDProducerWrapperT[T: Typeable & EDProducer](Movable, Typeable):
    var _ptr: UnsafePointer[Self.T]

    @always_inline
    def __init__(out self, mut reg: ProductRegistry):
        self._ptr = UnsafePointer[T].alloc(1)
        __get_address_as_uninit_lvalue(self._ptr.address) = Self.T.__init__(reg)

    @always_inline
    def delete(self):
        self._ptr.destroy_pointee()
        self._ptr.free()

    @always_inline
    def __moveinit__(out self, var other: Self):
        self._ptr = other._ptr

    @always_inline
    def producer(self) -> UnsafePointer[Self.T]:
        return self._ptr

    @staticmethod
    @always_inline
    def dtype() -> String:
        return "EDProducerWrapperT[" + Self.T.dtype() + "]"


struct EDProducerConcrete(Copyable, Movable, Typeable):
    comptime _C = def (mut ProductRegistry) raises -> EDProducerWrapper
    comptime _P = def (mut EDProducerWrapper, mut Event, EventSetup)
    comptime _E = def (mut EDProducerWrapper) raises
    comptime _D = def (mut EDProducerWrapper)
    var _producer: EDProducerWrapper
    var _create: Self._C
    var _produce: Self._P
    var _end: Self._E
    var _det: Self._D

    @always_inline
    def __init__(
        out self, create: Self._C, produce: Self._P, end: Self._E, det: Self._D
    ):
        self._producer = EDProducerWrapper()
        self._create = create
        self._produce = produce
        self._end = end
        self._det = det

    @always_inline
    def __copyinit__(out self, other: Self):
        self._producer = other._producer
        self._create = other._create
        self._produce = other._produce
        self._end = other._end
        self._det = other._det

    @always_inline
    def __moveinit__(out self, var other: Self):
        self._producer = other._producer^
        self._create = other._create
        self._produce = other._produce
        self._end = other._end
        self._det = other._det

    @always_inline
    def create(mut self, mut reg: ProductRegistry) raises:
        self._producer = self._create(reg)

    @always_inline
    def produce(mut self, mut event: Event, ref eventSetup: EventSetup):
        self._produce(self._producer, event, eventSetup)

    @always_inline
    def endJob(mut self) raises:
        self._end(self._producer)

    @always_inline
    def delete(mut self):
        self._det(self._producer)

    @staticmethod
    @always_inline
    def dtype() -> String:
        return "EDProducerConcrete"


struct Registry(Typeable):
    comptime _pluginRegistryType = Dict[String, EDProducerConcrete]
    var _pluginRegistry: Self._pluginRegistryType

    @always_inline
    def __init__(out self):
        self._pluginRegistry = {}

    @always_inline
    def __deinit__(var self):
        self.delete()

    @always_inline
    def __getitem__(self, var name: String) raises -> EDProducerConcrete:
        return self._pluginRegistry[name^]

    @always_inline
    def __setitem__(
        mut self, var name: String, var esproducer: EDProducerConcrete
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
struct PluginFactory:
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
        var name: String, mut preg: ProductRegistry, mut reg: Registry
    ) raises -> EDProducerConcrete:
        reg[name].create(preg)
        return reg[name^]


@always_inline
def fwkModule[T: Typeable & EDProducer](mut reg: Registry):
    @always_inline
    def create_templ[
        T: Typeable & EDProducer
    ](mut reg: ProductRegistry) raises -> EDProducerWrapper:
        return rebind[EDProducerWrapper](EDProducerWrapperT[T](reg))

    @always_inline
    def produce_templ[
        T: Typeable & EDProducer
    ](
        mut edproducer: EDProducerWrapper,
        mut event: Event,
        eventSetup: EventSetup,
    ):
        rebind[EDProducerWrapperT[T]](edproducer).producer()[].produce(
            event, eventSetup
        )

    @always_inline
    def end_templ[T: Typeable & EDProducer](mut edproducer: EDProducerWrapper) raises:
        rebind[EDProducerWrapperT[T]](edproducer).producer()[].endJob()

    @always_inline
    def det_templ[T: Typeable & EDProducer](mut edproducer: EDProducerWrapper):
        rebind[EDProducerWrapperT[T]](edproducer).delete()

    var crp = EDProducerConcrete(
        create_templ[T], produce_templ[T], end_templ[T], det_templ[T]
    )
    try:
        reg[T.dtype()] = crp^
    except e:
        print(
            "Framework/EDPluginFactory.mojo, failed to register plugin ",
            T.dtype(),
            ", got error: ",
            e,
            sep="",
        )
