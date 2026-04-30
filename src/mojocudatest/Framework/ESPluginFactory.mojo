from std.memory import OwnedPointer
from memory import Arc
from Framework.ESProducer import ESProducer


trait ESProducerConstructible:
    fn __init__(out self, var datadir: String):
        ...

# Define the trait (replaces the abstract base class)
trait MakerBase:
    def create(self, datadir: String) -> OwnedPointer[ESProducer]: ...

# Generic struct that conforms to the trait
@fieldwise_init
struct Maker[T:ESProducer & ESProducerConstructible](MakerBase):
    def create(self, datadir: String) -> OwnedPointer[ESProducer]:
        return OwnedPointer[ESProducer](T(datadir))




struct Registry(Movable):
    # couldnt use OwnedPointer here because we want to be able to move the registry itself, and OwnedPointer is not movable
    var pluginRegistry_: Dict[String, Arc[MakerBase]]

    fn __init__(out self):
        self.pluginRegistry_ = Dict[String, Arc[MakerBase]]()

    fn __moveinit__(out self, var other: Self):
        self.pluginRegistry_ = other.pluginRegistry_^

    fn add(mut self, name: String, maker: Arc[MakerBase]) raises:
        if name in self.pluginRegistry_:
            raise "RuntimeError: Plugin " + name + " is already registered"
        self.pluginRegistry_[name] = maker

    fn get(self, name: String) raises -> ref [self.pluginRegistry_] MakerBase:
        if name not in self.pluginRegistry_:
            raise "RuntimeError: Plugin " + name + " is not registered"
        return self.pluginRegistry_[name][]



var _global_registry = Registry()

def get_global_registry() -> ref [_global_registry] Registry:
    return _global_registry


@nonmaterializable(NoneType)
struct ESPluginFactory:
    # Port of edm::ESPluginFactory::create(...)
    @staticmethod
    fn create(
        name: String, datadir: String
    ) raises -> OwnedPointer[ESProducer]:
        return get_global_registry().get(name).create(datadir)
