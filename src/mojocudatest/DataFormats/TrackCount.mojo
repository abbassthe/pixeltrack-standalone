from MojoBridge.DTypes import Typeable


@fieldwise_init
@register_passable("trivial")
struct TrackCount(Copyable, Defaultable, Movable, Typeable):
    var _tracks: UInt32

    @always_inline
    fn __init__(out self):
        self._tracks = 0

    @always_inline
    fn nTracks(self) -> UInt32:
        return self._tracks

    @always_inline
    @staticmethod
    fn dtype() -> String:
        return "TrackCount"
