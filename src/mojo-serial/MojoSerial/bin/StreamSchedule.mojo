from std.collections import Deque

from MojoSerial.Framework.Event import Event
from MojoSerial.Framework.EventSetup import EventSetup
from MojoSerial.Framework.ProductRegistry import ProductRegistry
from MojoSerial.Framework.PluginFactory import PluginFactory, EDProducerConcrete
from MojoSerial.MojoBridge.DTypes import Typeable
from MojoSerial.bin.Source import Source


struct StreamSchedule(Defaultable, Movable, Typeable):
    var _source: UnsafePointer[Source]
    var _eventSetup: UnsafePointer[EventSetup]
    var _path: List[EDProducerConcrete]
    var _streamId: Int32

    @always_inline
    def __init__(out self):
        self._source = UnsafePointer[Source]()
        self._eventSetup = UnsafePointer[EventSetup]()
        self._path = []
        self._streamId = 0

    def __init__(
        out self,
        mut reg: ProductRegistry,
        source: UnsafePointer[Source],
        eventSetup: UnsafePointer[EventSetup],
        mut edreg: MojoSerial.Framework.PluginFactory.Registry,
        streamId: Int32 = 0,
    ):
        try:
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
                reg.beginModuleConstruction(i + 1)
                producers.append(
                    PluginFactory.create(name, reg, edreg)
                )
                var dep_indices = reg.consumedModules()
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
            print("Error occurred in bin/StreamSchedule.mojo,", e)
            return Self()

    @always_inline
    def __moveinit__(out self, var other: Self):
        self._source = other._source
        self._eventSetup = other._eventSetup
        self._path = other._path^
        self._streamId = other._streamId

    def run(mut self, ref reg: ProductRegistry):
        var event: Event
        var ptr = self._source[].produce(self._streamId, reg)
        while ptr != UnsafePointer[Event]():
            event = ptr.take_pointee()
            ptr.free()
            for i in range(self._path.__len__()):
                self._path[i].produce(event, self._eventSetup[])
            ptr = self._source[].produce(self._streamId, reg)

    def endJob(mut self) raises:
        for i in range(self._path.__len__()):
            self._path[i].endJob()

    @always_inline
    @staticmethod
    def dtype() -> String:
        return "StreamSchedule"
