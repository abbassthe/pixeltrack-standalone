from std.sys import size_of
import std.os as os


@always_inline
def read_simd[T: DType](mut file: FileHandle) raises -> Scalar[T]:
    var obj = file.read_bytes(size_of[T]())
    return obj.steal_data().bitcast[Scalar[T]]().take_pointee()


@always_inline
def read_simd_eof[
    T: DType
](mut file: FileHandle) raises -> Tuple[Bool, Scalar[T]]:
    var obj = file.read_bytes(size_of[T]())
    if obj.__len__() < size_of[T]():
        return True, 0
    return False, obj.steal_data().bitcast[Scalar[T]]().take_pointee()


@always_inline
def read_obj[T: Movable](mut file: FileHandle) raises -> T:
    var obj = file.read_bytes(size_of[T]())
    return obj.steal_data().bitcast[T]().take_pointee()


@always_inline
def read_list[
    T: Movable & Copyable
](mut file: FileHandle, var num: Int) raises -> List[T]:
    var ret = List[T](unsafe_uninit_length=num)
    var elements = file.read_bytes(num * size_of[T]())
    var data = elements.steal_data().bitcast[T]()
    for i in range(num):
        (data + i).move_pointee_into(ret._data + i)
    return ret^
