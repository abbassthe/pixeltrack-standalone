from std.memory import OwnedPointer
from MojoSerial.MojoBridge.DTypes import SizeType, Typeable


@fieldwise_init
struct DeviceConstView(Copyable, Defaultable, Movable, Typeable, TrivialRegisterPassable):
    var _moduleStart: UnsafePointer[UInt32]
    var _clusInModule: UnsafePointer[UInt32]
    var _moduleId: UnsafePointer[UInt32]
    var _clusModuleStart: UnsafePointer[UInt32]

    @always_inline
    def __init__(out self):
        self._moduleStart = UnsafePointer[UInt32]()
        self._clusInModule = UnsafePointer[UInt32]()
        self._moduleId = UnsafePointer[UInt32]()
        self._clusModuleStart = UnsafePointer[UInt32]()

    @always_inline
    def moduleStart(self, i: Int) -> UInt32:
        return self._moduleStart[i]

    @always_inline
    def clusInModule(self, i: Int) -> UInt32:
        return self._clusInModule[i]

    @always_inline
    def moduleId(self, i: Int) -> UInt32:
        return self._moduleId[i]

    @always_inline
    def clusModuleStart(self, i: Int) -> UInt32:
        return self._clusModuleStart[i]

    @always_inline
    @staticmethod
    def dtype() -> String:
        return "DeviceConstView"


struct SiPixelClustersSoA(Defaultable, Movable, Typeable):
    var moduleStart_d: OwnedPointer[
        List[UInt32]
    ]  # index of the first pixel of each module
    var clusInModule_d: OwnedPointer[
        List[UInt32]
    ]  # number of clusters found in each module
    var moduleId_d: OwnedPointer[List[UInt32]]  # module id of each module
    var clusModuleStart_d: OwnedPointer[
        List[UInt32]
    ]  # index of the first cluster of each module
    var view_d: OwnedPointer[DeviceConstView]  # "me" pointer
    var nClusters_h: UInt32

    def __init__(out self):
        self.moduleStart_d = OwnedPointer(List[UInt32]())
        self.clusInModule_d = OwnedPointer(List[UInt32]())
        self.moduleId_d = OwnedPointer(List[UInt32]())
        self.clusModuleStart_d = OwnedPointer(List[UInt32]())
        self.view_d = OwnedPointer(
            DeviceConstView(
                self.moduleStart_d[].unsafe_ptr(),
                self.clusInModule_d[].unsafe_ptr(),
                self.moduleId_d[].unsafe_ptr(),
                self.clusModuleStart_d[].unsafe_ptr(),
            )
        )
        self.nClusters_h = 0

    def __init__(out self, maxClusters: SizeType):
        debug_assert(maxClusters >= 0)
        self.moduleStart_d = OwnedPointer(
            List[UInt32](length=UInt(maxClusters) + 1, fill=0)
        )
        self.clusInModule_d = OwnedPointer(
            List[UInt32](length=UInt(maxClusters), fill=0)
        )
        self.moduleId_d = OwnedPointer(
            List[UInt32](length=UInt(maxClusters), fill=0)
        )
        self.clusModuleStart_d = OwnedPointer(
            List[UInt32](length=UInt(maxClusters) + 1, fill=0)
        )
        self.view_d = OwnedPointer(
            DeviceConstView(
                self.moduleStart_d[].unsafe_ptr(),
                self.clusInModule_d[].unsafe_ptr(),
                self.moduleId_d[].unsafe_ptr(),
                self.clusModuleStart_d[].unsafe_ptr(),
            )
        )
        self.nClusters_h = 0

    def __moveinit__(out self, var other: Self):
        self.moduleStart_d = other.moduleStart_d^
        self.clusInModule_d = other.clusInModule_d^
        self.moduleId_d = other.moduleId_d^
        self.clusModuleStart_d = other.clusModuleStart_d^
        self.nClusters_h = other.nClusters_h
        self.view_d = OwnedPointer(
            DeviceConstView(
                self.moduleStart_d[].unsafe_ptr(),
                self.clusInModule_d[].unsafe_ptr(),
                self.moduleId_d[].unsafe_ptr(),
                self.clusModuleStart_d[].unsafe_ptr(),
            )
        )

    def view(self) -> UnsafePointer[DeviceConstView, mut=False]:
        return self.view_d.unsafe_ptr()

    @always_inline
    def nClusters(self) -> UInt32:
        return self.nClusters_h

    @always_inline
    def setNClusters(mut self, nClusters: UInt32):
        self.nClusters_h = nClusters

    @always_inline
    def moduleStart[
        origin: Origin, //
    ](ref [origin]self) -> UnsafePointer[
        UInt32, mut = origin.mut, origin=origin
    ]:
        return self.moduleStart_d[].unsafe_ptr()

    @always_inline
    def clusInModule[
        origin: Origin, //
    ](ref [origin]self) -> UnsafePointer[
        UInt32, mut = origin.mut, origin=origin
    ]:
        return self.clusInModule_d[].unsafe_ptr()

    @always_inline
    def moduleId[
        origin: Origin, //
    ](ref [origin]self) -> UnsafePointer[
        UInt32, mut = origin.mut, origin=origin
    ]:
        return self.moduleId_d[].unsafe_ptr()

    @always_inline
    def clusModuleStart[
        origin: Origin, //
    ](ref [origin]self) -> UnsafePointer[
        UInt32, mut = origin.mut, origin=origin
    ]:
        return self.clusModuleStart_d[].unsafe_ptr()

    @always_inline
    def c_moduleStart(self) -> UnsafePointer[UInt32, mut=False]:
        return self.moduleStart_d[].unsafe_ptr()

    @always_inline
    def c_clusInModule(self) -> UnsafePointer[UInt32, mut=False]:
        return self.clusInModule_d[].unsafe_ptr()

    @always_inline
    def c_moduleId(self) -> UnsafePointer[UInt32, mut=False]:
        return self.moduleId_d[].unsafe_ptr()

    @always_inline
    def c_clusModuleStart(self) -> UnsafePointer[UInt32, mut=False]:
        return self.clusModuleStart_d[].unsafe_ptr()

    @always_inline
    @staticmethod
    def dtype() -> String:
        return "SiPixelClustersSoA"
