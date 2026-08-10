# Mojo port of CUDACore/cudastdAlgorithm.h. Only upper_bound is ported --
# the only one HistoContainer.mojo's kernels actually call. No custom
# Compare parameter (C++ defaults to cuda_std::less<T>, i.e. `<`) since
# nothing here supplies one either.
from std.sys.info import size_of


fn upper_bound[
    T: DType, //
](
    var first: UnsafePointer[Scalar[T], MutAnyOrigin],
    var last: UnsafePointer[Scalar[T], MutAnyOrigin],
    value: Scalar[T],
) -> UnsafePointer[Scalar[T], MutAnyOrigin]:
    var count = (Int(last) - Int(first)) // size_of[Scalar[T]]()

    while count > 0:
        var it = first
        var step = count // 2
        it += step

        if value >= it[]:
            it += 1
            first = it
            count -= step + 1
        else:
            count = step

    return first
