from memory import Arc
from os.atomic import Atomic

from MojoBridge.ConcurrentQueue import ConcurrentQueue

# -*- Mojo -*-
#
# Package:     FWCore/Utilities
# Class  :     ReusableObjectHolder
#
# \class edm::ReusableObjectHolder ReusableObjectHolder "ReusableObjectHolder.h"
#
# Description: Thread safe way to do create and reuse a group of the same object type.
#
# Usage:
# This class can be used to safely reuse a series of objects created on demand. The reuse
# of the objects is safe even across different threads since one can safely call all member
# functions of this class on the same instance of this class from multiple threads.
#
# This class manages the cache of reusable objects and therefore an instance of this
# class must live as long as you want the cache to live.
#
# The primary way of using the class it to call makeOrGetAndClear
# An example use would be
#
#   var objectToUse = holder.makeOrGetAndClear(
#                            fn() -> MyObject: return MyObject(10),   # makes new one
#                            fn(old: MyObject): old.reset()           # resets old one
#                    )
#
# If you always want to set the values you can use makeOrGet
#
#   var objectToUse = holder.makeOrGet(fn() -> MyObject: return MyObject())
#   objectToUse.setValue(3)
#
# NOTE: If you hold onto the Arc until another call to the ReusableObjectHolder,
# make sure to release the Arc before the call. That way the object you were just
# using can go back into the cache and be reused for the call you are going to make.
# An example
#
#   while someCondition():
#       var obj = holder.tryToGet()
#       obj.value().setValue(someNewValue())
#       useTheObject(obj.value())
#       # obj goes out of scope, Arc ref count drops to zero,
#       # _HeldItem.__del__ fires and returns the item to the cache
#
#
# Original Author:  Chris Jones
#         Created:  Fri, 31 July 2014 14:29:41 GMT
#
# Ported to Mojo by Abbas Naim
#


# Equivalent to the shared_ptr<T> with custom deleter returned by C++ tryToGet.
# When the last Arc reference drops, __del__ fires and returns the item to the
# pool via the back-pointer, mirroring the lambda in addBack.
struct _HeldItem[T: Copyable & ImplicitlyDestructible](Movable):
    var _item: Optional[T]
    var _holder: UnsafePointer[ReusableObjectHolder[T]]

    fn __init__(
        out self,
        owned item: Optional[T],
        holder: UnsafePointer[ReusableObjectHolder[T]],
    ):
        self._item = item^
        self._holder = holder

    fn __moveinit__(out self, var other: Self):
        self._item = other._item^
        self._holder = other._holder

    fn __del__(owned self):
        # mirrors: pHolder->addBack(std::unique_ptr<T,Deleter>{iItem, deleter})
        # Empty Arc (null item) does nothing, mirroring empty shared_ptr destructor.
        if self._item:
            self._holder[]._addBack(self._item.value()^)


struct ReusableObjectHolder[T: Copyable & ImplicitlyDestructible](Movable):
    var m_availableQueue: ConcurrentQueue[T]
    var m_outstandingObjects: Atomic[DType.int64]

    fn __init__(out self):
        self.m_availableQueue = ConcurrentQueue[T]()
        self.m_outstandingObjects = Atomic[DType.int64](0)

    fn __moveinit__(out self, var other: Self):
        debug_assert(other.m_outstandingObjects.load() == 0, "ReusableObjectHolder moved while items are still outstanding")
        self.m_availableQueue = other.m_availableQueue^
        self.m_outstandingObjects = Atomic[DType.int64](0)

    fn __del__(owned self):
        debug_assert(self.m_outstandingObjects.load() == 0, "ReusableObjectHolder destroyed while items are still outstanding")
        # m_availableQueue drops here automatically, destroying all held items

    # Adds the item to the cache.
    # Use this function if you know ahead of time
    # how many cached items you will need.
    fn add(mut self, owned item: T):
        self.m_availableQueue.enqueue(item^)

    # Tries to get an already created object.
    # If none are available, returns None (equivalent to empty shared_ptr<T>{}).
    # If one is available, returns Arc[_HeldItem[T]] whose destructor automatically
    # returns the item to the pool — equivalent to shared_ptr<T> with custom deleter.
    # Use this function in conjunction with add().
    fn tryToGet(mut self) -> Arc[_HeldItem[T]]:
        var item = self.m_availableQueue.dequeue()
        if item:
            _ = self.m_outstandingObjects.fetch_add(1)
            return Arc[_HeldItem[T]](
                _HeldItem[T](item^, UnsafePointer.address_of(self))
            )
        # Empty Arc — equivalent to std::shared_ptr<T>{}
        return Arc[_HeldItem[T]](
            _HeldItem[T](None, UnsafePointer[ReusableObjectHolder[T]]())
        )
    
    fn makeOrGetAndClear[FM , FC](mut self,iMakeFunc : FM, iClearFunc: FC) -> Arc[_HeldItem[T]]
        where FM: Fn() -> T, FC: Fn(T) -> Void:
        var returnValue: Arc[_HeldItem[T]] = self.tryToGet()
        while not returnValue:
            self.add(makeUnique(iMakeFunc()))
            returnValue = self.tryToGet()

        iClearFunc(returnValue[]._item.value())
        return returnValue

    fn makeOrGet[FM: Fn() -> T](mut self, iMakeFunc: FM) -> Arc[_HeldItem[T]]:
        var returnValue = self.tryToGet()
        while not returnValue[]._item:
            self.add(iMakeFunc())
            returnValue = self.tryToGet()
        return returnValue

    # Private — mirrors C++ addBack, called only by _HeldItem.__del__.
    fn _addBack(mut self, owned item: T):
        self.m_availableQueue.enqueue(item^)
        _ = self.m_outstandingObjects.fetch_sub(1)

    # Mirrors: std::unique_ptr<T> makeUnique(T* ptr)
    # C++ has static_assert(is_same_v<Deleter, default_delete<T>>) here,
    # guarding against calling this overload when a custom Deleter is in use.
    # Mojo has no Deleter parameter so the assert has no equivalent to port.
    fn _makeUnique(owned item: T) -> Arc[T]:
        return Arc[T](item^)

    # Passes an already-wrapped Arc straight through.
    # Mirrors: std::unique_ptr<T, Deleter> makeUnique(std::unique_ptr<T, Deleter> ptr)
    fn _makeUnique(owned arc: Arc[T]) -> Arc[T]:
        return arc^

    # Remove all idle items from the pool.
    fn clear(mut self):
        while self.m_availableQueue.dequeue():
            pass
