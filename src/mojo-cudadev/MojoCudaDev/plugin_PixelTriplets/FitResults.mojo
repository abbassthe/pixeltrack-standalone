# Mojo port of plugin-PixelTriplets/FitResult.h.
from MojoCudaDev.MojoBridge.Matrix import Matrix

comptime Vector2d = Matrix[DType.float64, 2, 1]
comptime Vector3d = Matrix[DType.float64, 3, 1]
comptime Vector4d = Matrix[DType.float64, 4, 1]
comptime Vector5d = Matrix[DType.float64, 5, 1]
comptime Matrix2d = Matrix[DType.float64, 2, 2]
comptime Matrix3d = Matrix[DType.float64, 3, 3]
comptime Matrix4d = Matrix[DType.float64, 4, 4]
comptime Matrix5d = Matrix[DType.float64, 5, 5]
comptime Matrix6d = Matrix[DType.float64, 6, 6]

comptime Matrix3xNd[N: Int] = Matrix[DType.float64, 3, N]  # used for inputs hits


@fieldwise_init
struct _CircleFit(Copyable, Defaultable, Movable):
    var par: Vector3d  #!< parameter: (X0,Y0,R)
    var cov: Matrix3d

    #/*!< covariance matrix: \n
    #  |cov(X0,X0)|cov(Y0,X0)|cov( R,X0)| \n
    #  |cov(X0,Y0)|cov(Y0,Y0)|cov( R,Y0)| \n
    #  |cov(X0, R)|cov(Y0, R)|cov( R, R)|
    #*/

    var qCharge: Int32  #!< particle charge
    var chi2: Float32

    fn __init__(out self):
        self.par = Vector3d()
        self.cov = Matrix3d()
        self.qCharge = 0
        self.chi2 = 0.0


@fieldwise_init
struct _LineFit(Copyable, Defaultable, Movable):
    var par: Vector2d  #!<(cotan(theta),Zip)
    var cov: Matrix2d

    #/*!<
    #  |cov(c_t,c_t)|cov(Zip,c_t)| \n
    #  |cov(c_t,Zip)|cov(Zip,Zip)|
    #*/

    var chi2: Float64

    fn __init__(out self):
        self.par = Vector2d()
        self.cov = Matrix2d()
        self.chi2 = 0.0


@fieldwise_init
struct _HelixFit(Copyable, Defaultable, Movable):
    var par: Vector5d  #!<(phi,Tip,pt,cotan(theta)),Zip)
    var cov: Matrix5d

    #/*!< ()->cov() \n
    #  |(phi,phi)|(Tip,phi)|(p_t,phi)|(c_t,phi)|(Zip,phi)| \n
    #  |(phi,Tip)|(Tip,Tip)|(p_t,Tip)|(c_t,Tip)|(Zip,Tip)| \n
    #  |(phi,p_t)|(Tip,p_t)|(p_t,p_t)|(c_t,p_t)|(Zip,p_t)| \n
    #  |(phi,c_t)|(Tip,c_t)|(p_t,c_t)|(c_t,c_t)|(Zip,c_t)| \n
    #  |(phi,Zip)|(Tip,Zip)|(p_t,Zip)|(c_t,Zip)|(Zip,Zip)|
    #*/

    var chi2_circle: Float32
    var chi2_line: Float32
    #    Vector4d fast_fit;
    var qCharge: Int32  #!< particle charge

    fn __init__(out self):
        self.par = Vector5d()
        self.cov = Matrix5d()
        self.chi2_circle = 0.0
        self.chi2_line = 0.0
        self.qCharge = 0