from collections import Deque

from Framework.Event import Event
from Framework.EventSetup import EventSetup
from Framework.ProductRegistry import ProductRegistry
from Framework.PluginFactory import PluginFactory, EDProducerConcrete
from MojoBridge.DTypes import Typeable
from Bin.Source import Source


struct StreamSchedule(Defaultable, Movable, Typeable):
    var _registry: UnsafePointer[ProductRegistry]
    var _source: UnsafePointer[Source]
    var _eventSetup: UnsafePointer[EventSetup]
    var _path: List[EDProducerConcrete]
    var _streamId: Int32

    @always_inline
    fn __init__(out self):
        self._registry = UnsafePointer[ProductRegistry]()
        self._source = UnsafePointer[Source]()
        self._eventSetup = UnsafePointer[EventSetup]()
        self._path = []
        self._streamId = 0

    fn __init__(
        out self,
        reg: UnsafePointer[ProductRegistry],
        source: UnsafePointer[Source],
        eventSetup: UnsafePointer[EventSetup],
        mut edreg: Framework.PluginFactory.Registry,
        streamId: Int32 = 0,
    ):
        try:
            self._registry = reg
            self._source = source
            self._eventSetup = eventSetup
            self._streamId = streamId

            var nModules = PluginFactory.size(edreg)
            debug_assert(nModules > 0)

            var producers = List[EDProducerConcrete](capacity=nModules)
            var adj = List[List[Int]](length=nModules, fill=[])
            var in_degree = List[Int](length=nModules, fill=0)

            var i: UInt = 0
            for name in PluginFactory.getAll(edreg):
                self._registry[].beginModuleConstruction(i + 1)
                producers.append(
                    PluginFactory.create(name, self._registry[], edreg)
                )
                var dep_indices = self._registry[].consumedModules()
                # remove dependency on FEDRawDataCollection from resolver logic
                # it is the parent of all producers (guaranteed by Source)
                if 0 in dep_indices:
                    dep_indices.remove(0)
                in_degree[i] = dep_indices.__len__()

                for dep_index in dep_indices:
                    adj[dep_index - 1].append(i)
                i += 1

            var q = Deque[Int]()
            for i in range(nModules):
                if in_degree[i] == 0:
                    q.append(i)

            var sorted_indices = List[Int](capacity=nModules)
            while q.__len__() > 0:
                var u = q.pop()
                sorted_indices.append(u)

                for v in adj[u]:
                    in_degree[v] -= 1
                    if in_degree[v] == 0:
                        q.append(v)

            if sorted_indices.__len__() != nModules:
                raise Error(
                    "A cycle was detected in the module dependency graph."
                )

            self._path = List[EDProducerConcrete](capacity=nModules)
            var data = producers.steal_data()
            for index in sorted_indices:
                self._path.append((data + index).take_pointee())
        except e:
            print("Error occurred in Bin/StreamSchedule.mojo,", e)
            return Self()

    @always_inline
    fn __moveinit__(out self, var other: Self):
        self._registry = other._registry
        self._source = other._source
        self._eventSetup = other._eventSetup
        self._path = other._path^
        self._streamId = other._streamId

    fn run(mut self):
        var event: Event
        var ptr = self._source[].produce(self._streamId, self._registry[])
        while ptr != UnsafePointer[Event]():
            event = ptr.take_pointee()
            ptr.free()
            for i in range(self._path.__len__()):
                self._path[i].produce(event, self._eventSetup[])
            ptr = self._source[].produce(self._streamId, self._registry[])

    fn endJob(mut self):
        for i in range(self._path.__len__()):
            self._path[i].endJob()

    @always_inline
    @staticmethod
    fn dtype() -> String:
        return "StreamSchedule"
