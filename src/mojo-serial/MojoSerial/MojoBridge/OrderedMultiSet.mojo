@fieldwise_init
struct OrderedMultiSet[T: Copyable, Compare: AnyType](Movable, Sized):
    # Sorted storage using a comparator type with `less(a, b) -> Bool`.
    var _items: List[Self.T]

    @always_inline
    def __init__(out self):
        self._items = List[Self.T]()

    @always_inline
    def __len__(self) -> Int:
        return self._items.__len__()

    @always_inline
    def is_empty(self) -> Bool:
        return self._items.__len__() == 0

    @always_inline
    def clear(ref self):
        self._items.clear()

    @always_inline
    def __getitem__(self, index: Int) raises -> Self.T:
        return self._items[index]

    @always_inline
    def _less(self, a: read T, b: read T) -> Bool:
        return Compare.less(a, b)

    @always_inline
    def _equivalent(self, a: read T, b: read T) -> Bool:
        return (not self._less(a, b)) and (not self._less(b, a))

    # Returns the first position where `value` can be inserted without
    # violating sorted order.
    def lower_bound(self, value: read T) -> Int:
        var i: Int = 0
        var n = self._items.__len__()
        while i < n:
            if not self._less(self._items[i], value):
                return i
            i += 1
        return n

    # Returns one matching index or -1 if no equivalent element exists.
    def find(self, value: read T) -> Int:
        var i = self.lower_bound(value)
        if i < self._items.__len__() and self._equivalent(self._items[i], value):
            return i
        return -1

    def insert(mut self, value: Self.T):
        # Append and shift left to keep ordering by comparator.
        self._items.append(value)
        var i = self._items.__len__() - 1
        while i > 0 and self._less(self._items[i], self._items[i - 1]):
            var tmp = self._items[i - 1]
            self._items[i - 1] = self._items[i]
            self._items[i] = tmp
            i -= 1

    def erase_at(mut self, index: Int) -> Bool:
        if index < 0 or index >= self._items.__len__():
            return False
        self._items.remove(index)
        return True

    def erase_one(mut self, value: read T) -> Bool:
        var index = self.find(value)
        if index == -1:
            return False
        self._items.remove(index)
        return True
