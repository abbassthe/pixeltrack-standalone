def blockPrefixScan[
    VT: DType
](
    ci: UnsafePointer[Scalar[VT]],
    co: UnsafePointer[Scalar[VT], mut=True],
    size: UInt32,
):
    co[0] = ci[0]
    for i in range(1, size):
        co[i] = ci[i] + co[i - 1]


def blockPrefixScan[
    VT: DType
](c: UnsafePointer[Scalar[VT], mut=True], size: UInt32):
    for i in range(1, size):
        c[i] += c[i - 1]


# multiBlockPrefixScan is actually a non-working stub, ignore
