from MojoCudaDev.MojoBridge.DTypes import Typeable


@fieldwise_init
@register_passable("trivial")
struct TkRotation[T: DType](Copyable, Defaultable, Movable, Typeable):
    @always_inline
    @staticmethod
    fn dtype() -> String:
        return "TkRotation"


@fieldwise_init
@register_passable("trivial")
struct SOARotation[T: DType](Copyable, Defaultable, Movable):
    var R11: Scalar[T]
    var R12: Scalar[T]
    var R13: Scalar[T]
    var R21: Scalar[T]
    var R22: Scalar[T]
    var R23: Scalar[T]
    var R31: Scalar[T]
    var R32: Scalar[T]
    var R33: Scalar[T]

    @always_inline
    fn __init__(out self):
        self.R11 = 0
        self.R12 = 0
        self.R13 = 0
        self.R21 = 0
        self.R22 = 0
        self.R23 = 0
        self.R31 = 0
        self.R32 = 0
        self.R33 = 0

    @always_inline
    fn __init__(out self, var x: Scalar[T], /):
        self.R11 = 1
        self.R12 = 0
        self.R13 = 0
        self.R21 = 0
        self.R22 = 1
        self.R23 = 0
        self.R31 = 0
        self.R32 = 0
        self.R33 = 1

    @always_inline
    fn __init__(out self, p: UnsafePointer[Scalar[T]]):
        self.R11 = p[0]
        self.R12 = p[1]
        self.R13 = p[2]
        self.R21 = p[3]
        self.R22 = p[4]
        self.R23 = p[5]
        self.R31 = p[6]
        self.R32 = p[7]
        self.R33 = p[8]

    @always_inline
    fn transposed(self) -> Self:
        return Self(
            self.R11,
            self.R21,
            self.R31,
            self.R12,
            self.R22,
            self.R32,
            self.R13,
            self.R23,
            self.R33,
        )

    @always_inline
    fn multiply(
        self,
        var vx: Scalar[T],
        var vy: Scalar[T],
        var vz: Scalar[T],
        mut ux: Scalar[T],
        mut uy: Scalar[T],
        mut uz: Scalar[T],
    ):
        ux = self.R11 * vx + self.R12 * vy + self.R13 * vz
        uy = self.R21 * vx + self.R22 * vy + self.R23 * vz
        uz = self.R31 * vx + self.R32 * vy + self.R33 * vz

    @always_inline
    fn multiplyInverse(
        self,
        var vx: Scalar[T],
        var vy: Scalar[T],
        var vz: Scalar[T],
        mut ux: Scalar[T],
        mut uy: Scalar[T],
        mut uz: Scalar[T],
    ):
        ux = self.R11 * vx + self.R21 * vy + self.R31 * vz
        uy = self.R12 * vx + self.R22 * vy + self.R32 * vz
        uz = self.R13 * vx + self.R23 * vy + self.R33 * vz

    @always_inline
    fn multiplyInverse(
        self,
        var vx: Scalar[T],
        var vy: Scalar[T],
        mut ux: Scalar[T],
        mut uy: Scalar[T],
        mut uz: Scalar[T],
    ):
        ux = self.R11 * vx + self.R21 * vy
        uy = self.R12 * vx + self.R22 * vy
        uz = self.R13 * vx + self.R23 * vy

    @always_inline
    fn xx(self) -> Scalar[T]:
        return self.R11

    @always_inline
    fn xy(self) -> Scalar[T]:
        return self.R12

    @always_inline
    fn xz(self) -> Scalar[T]:
        return self.R13

    @always_inline
    fn yx(self) -> Scalar[T]:
        return self.R21

    @always_inline
    fn yy(self) -> Scalar[T]:
        return self.R22

    @always_inline
    fn yz(self) -> Scalar[T]:
        return self.R23

    @always_inline
    fn zx(self) -> Scalar[T]:
        return self.R31

    @always_inline
    fn zy(self) -> Scalar[T]:
        return self.R32

    @always_inline
    fn zz(self) -> Scalar[T]:
        return self.R33

    @always_inline
    @staticmethod
    fn dtype() -> String:
        return "SOARotation[" + T.__repr__() + "]"


@fieldwise_init
@register_passable("trivial")
struct SOAFrame[T: DType](Copyable, Defaultable, Movable, Typeable):
    var px: Scalar[T]
    var py: Scalar[T]
    var pz: Scalar[T]
    var rot: SOARotation[T]

    @always_inline
    fn __init__(out self):
        self.px = 0
        self.py = 0
        self.pz = 0
        self.rot = SOARotation[T]()

    @always_inline
    fn rotation(self) -> SOARotation[T]:
        return self.rot

    @always_inline
    fn toLocal(
        self,
        var vx: Scalar[T],
        var vy: Scalar[T],
        var vz: Scalar[T],
        mut ux: Scalar[T],
        mut uy: Scalar[T],
        mut uz: Scalar[T],
    ):
        self.rot.multiply(vx - self.px, vy - self.py, vz - self.pz, ux, uy, uz)

    @always_inline
    fn toGlobal(
        self,
        var vx: Scalar[T],
        var vy: Scalar[T],
        var vz: Scalar[T],
        mut ux: Scalar[T],
        mut uy: Scalar[T],
        mut uz: Scalar[T],
    ):
        self.rot.multiplyInverse(vx, vy, vz, ux, uy, uz)
        ux += self.px
        uy += self.py
        uz += self.pz

    @always_inline
    fn toGlobal(
        self,
        var vx: Scalar[T],
        var vy: Scalar[T],
        mut ux: Scalar[T],
        mut uy: Scalar[T],
        mut uz: Scalar[T],
    ):
        self.rot.multiplyInverse(vx, vy, ux, uy, uz)
        ux += self.px
        uy += self.py
        uz += self.pz

    @always_inline
    fn toGlobal(
        self,
        var cxx: Scalar[T],
        var cxy: Scalar[T],
        var cyy: Scalar[T],
        gl: UnsafePointer[Scalar[T]],
    ):
        var r = self.rot

        gl[0] = r.xx() * (r.xx() * cxx + r.yx() * cxy) + r.yx() * (
            r.xx() * cxy + r.yx() * cyy
        )
        gl[1] = r.xx() * (r.xy() * cxx + r.yy() * cxy) + r.yx() * (
            r.xy() * cxy + r.yy() * cyy
        )
        gl[2] = r.xy() * (r.xy() * cxx + r.yy() * cxy) + r.yy() * (
            r.xy() * cxy + r.yy() * cyy
        )
        gl[3] = r.xx() * (r.xz() * cxx + r.yz() * cxy) + r.yx() * (
            r.xz() * cxy + r.yz() * cyy
        )
        gl[4] = r.xy() * (r.xz() * cxx + r.yz() * cxy) + r.yy() * (
            r.xz() * cxy + r.yz() * cyy
        )
        gl[5] = r.xz() * (r.xz() * cxx + r.yz() * cxy) + r.yz() * (
            r.xz() * cxy + r.yz() * cyy
        )

    @always_inline
    fn toLocal(
        self,
        ge: UnsafePointer[Scalar[T]],
        mut lxx: Scalar[T],
        mut lxy: Scalar[T],
        mut lyy: Scalar[T],
    ):
        var r = self.rot
        var cxx = ge[0]
        var cyx = ge[1]
        var cyy = ge[2]
        var czx = ge[3]
        var czy = ge[4]
        var czz = ge[5]

        lxx = (
            r.xx() * (r.xx() * cxx + r.xy() * cyx + r.xz() * czx)
            + r.xy() * (r.xx() * cyx + r.xy() * cyy + r.xz() * czy)
            + r.xz() * (r.xx() * czx + r.xy() * czy + r.xz() * czz)
        )

        lxy = (
            r.yx() * (r.xx() * cxx + r.xy() * cyx + r.xz() * czx)
            + r.yy() * (r.xx() * cyx + r.xy() * cyy + r.xz() * czy)
            + r.yz() * (r.xx() * czx + r.xy() * czy + r.xz() * czz)
        )

        lyy = (
            r.yx() * (r.yx() * cxx + r.yy() * cyx + r.yz() * czx)
            + r.yy() * (r.yx() * cyx + r.yy() * cyy + r.yz() * czy)
            + r.yz() * (r.yx() * czx + r.yy() * czy + r.yz() * czz)
        )

    @always_inline
    fn x(self) -> Scalar[T]:
        return self.px

    @always_inline
    fn y(self) -> Scalar[T]:
        return self.py

    @always_inline
    fn z(self) -> Scalar[T]:
        return self.pz

    @always_inline
    @staticmethod
    fn dtype() -> String:
        return "SOAFrame[" + T.__repr__() + "]"
