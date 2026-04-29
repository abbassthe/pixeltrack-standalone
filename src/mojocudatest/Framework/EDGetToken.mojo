from MojoBridge.DTypes import Typeable


@register_passable("trivial")
struct EDGetTokenT[T: Typeable](Copyable, Defaultable, Movable, Typeable):
    alias s_uninitializedValue = 0xFFFFFFFF
    var m_value: UInt

    fn __init__(out self):
        self.m_value = Self.s_uninitializedValue

    @always_inline
    fn __init__(out self, var iOther: EDGetToken):
        self.m_value = iOther.m_value

    @always_inline
    fn __init__[O: Typeable](out self, iValue: UInt):
        constrained[
            O.dtype() == "ProductRegistry",
            "Only the product registry can hand tokens",
        ]()
        self.m_value = iValue

    @always_inline
    fn index(self) -> UInt:
        return self.m_value

    @always_inline
    fn isUninitialized(self) -> Bool:
        return self.m_value == Self.s_uninitializedValue

    @staticmethod
    @always_inline
    fn dtype() -> String:
        return "EDGetTokenT[" + T.dtype() + "]"


@register_passable("trivial")
struct EDGetToken(Copyable, Defaultable, Movable, Typeable):
    alias s_uninitializedValue = 0xFFFFFFFF
    var m_value: UInt

    @always_inline
    fn __init__(out self):
        self.m_value = Self.s_uninitializedValue

    @always_inline
    fn __init__[T: Typeable, //](out self, var iOther: EDGetTokenT[T]):
        self.m_value = iOther.m_value

    @always_inline
    fn __init__[O: Typeable](out self, iValue: UInt):
        constrained[
            O.dtype() == "ProductRegistry",
            "Only the product registry can hand tokens",
        ]()
        self.m_value = iValue

    @always_inline
    fn index(self) -> UInt:
        return self.m_value

    @always_inline
    fn isUninitialized(self) -> Bool:
        return self.m_value == Self.s_uninitializedValue

    @staticmethod
    @always_inline
    fn dtype() -> String:
        return "EDGetToken"
