from collections import Deque
from utils.lock import BlockingSpinLock, BlockingScopedLock


struct ConcurrentQueue[ElementType: Copyable & ImplicitlyDestructible]:
    var _deque: Deque[ElementType]
    var _lock: BlockingSpinLock

    fn __init__(out self):
        self._deque = Deque[ElementType]()
        self._lock = BlockingSpinLock()

    fn enqueue(mut self, owned item: ElementType):
        with BlockingScopedLock(self._lock):
            self._deque.append(item)

    fn dequeue(mut self) -> Optional[ElementType]:
        with BlockingScopedLock(self._lock):
            if len(self._deque) == 0:
                return None
            return self._deque.popleft()

    fn peek_front(self) -> Optional[ElementType]:
        # peekleft() provides non-destructive access to the first element
        with BlockingScopedLock(self._lock):
            if len(self._deque) == 0:
                return None
            return self._deque.peekleft()

    fn __len__(self) -> Int:
        with BlockingScopedLock(self._lock):
            return len(self._deque)
