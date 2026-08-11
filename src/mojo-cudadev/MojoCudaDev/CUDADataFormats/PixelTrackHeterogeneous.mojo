# Mojo port of CUDADataFormats/PixelTrackHeterogeneous.h.
from MojoCudaDev.CUDADataFormats.HeterogeneousSoA import HeterogeneousSoA
from MojoCudaDev.CUDADataFormats.TrackSoAHeterogeneousT import pixelTrack

comptime PixelTrackHeterogeneous = HeterogeneousSoA[pixelTrack.TrackSoA]
