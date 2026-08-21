# Mojo port of CUDADataFormats/ZVertexHeterogeneous.h.
from MojoCudaDev.CUDACore.Product import Product
from MojoCudaDev.CUDADataFormats.HeterogeneousSoA import HeterogeneousSoA
from MojoCudaDev.CUDADataFormats.ZVertexSoA import ZVertexSoA

comptime ZVertexHeterogeneous = HeterogeneousSoA[ZVertexSoA]
comptime ZVertexCUDAProduct = Product[ZVertexHeterogeneous]
