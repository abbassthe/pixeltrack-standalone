# Mojo port of plugin-PixelTriplets/CAConstants.h.
from MojoCudaDev.CUDACore.OneToManyAssoc import OneToManyAssoc
from MojoCudaDev.CUDACore.SimpleVector import SimpleVector
from MojoCudaDev.CUDACore.VecArray import VecArray
from sys import is_defined


# Cellular automaton constants
struct caConstants:
    # constants
    comptime maxCellNeighbors: UInt32 = UInt32(
        64 if is_defined["ONLY_PHICUT"]() else 36
    )
    comptime maxCellTracks: UInt32 = UInt32(
        64 if is_defined["ONLY_PHICUT"]() else 48
    )
    comptime maxNumberOfTuples: UInt32 = UInt32(
        48 * 1024 if is_defined["ONLY_PHICUT"]() else (
            3 * 1024 if is_defined["GPU_SMALL_EVENTS"]() else 24 * 1024
        )
    )
    comptime maxNumberOfDoublets: UInt32 = UInt32(
        2 * 1024 * 1024 if is_defined["ONLY_PHICUT"]() else (
            128 * 1024 if is_defined["GPU_SMALL_EVENTS"]() else 512 * 1024
        )
    )
    comptime maxCellsPerHit: UInt32 = UInt32(
        8 * 128 if is_defined["ONLY_PHICUT"]() else (
            128 // 2 if is_defined["GPU_SMALL_EVENTS"]() else 128
        )
    )
    comptime maxNumOfActiveDoublets: UInt32 = Self.maxNumberOfDoublets // 8
    comptime maxNumberOfQuadruplets: UInt32 = Self.maxNumberOfTuples

    comptime maxNumberOfLayerPairs: UInt32 = 20
    comptime maxNumberOfLayers: UInt32 = 10
    comptime maxTuples: UInt32 = Self.maxNumberOfTuples

    # Modules constants
    comptime max_ladder_bpx0: UInt32 = 12
    comptime first_ladder_bpx0: UInt32 = 0
    comptime module_length_bpx0: Float32 = 6.7
    comptime module_tolerance_bpx0: Float32 = 0.4
    comptime max_ladder_bpx4: UInt32 = 64
    comptime first_ladder_bpx4: UInt32 = 84
    comptime radius_even_ladder: Float32 = 15.815
    comptime radius_odd_ladder: Float32 = 16.146
    comptime module_length_bpx4: Float32 = 6.7
    comptime module_tolerance_bpx4: Float32 = 0.2
    comptime barrel_z_length: Float32 = 26.0
    comptime forward_z_begin: Float32 = 32.0

    # Last indexes
    comptime last_bpix1_detIndex: UInt32 = 96
    comptime last_barrel_detIndex: UInt32 = 1184

    # types
    comptime hindex_type = UInt32
    comptime tindex_type = UInt16

    comptime CellNeighbors = VecArray[
        UInt32, "CellNeighbors", Int(Self.maxCellNeighbors)
    ]
    comptime CellTracks = VecArray[
        Self.tindex_type, "CellTracks", Int(Self.maxCellTracks)
    ]

    comptime CellNeighborsVector = SimpleVector[
        Self.CellNeighbors, "CellNeighborsVector"
    ]
    comptime CellTracksVector = SimpleVector[Self.CellTracks, "CellTracksVector"]

    comptime OuterHitOfCell = VecArray[
        UInt32, "OuterHitOfCell", Int(Self.maxCellsPerHit)
    ]
    comptime TuplesContainer = OneToManyAssoc[
        Self.hindex_type, Int(Self.maxTuples), Int(5 * Self.maxTuples)
    ]
    # 3.5 should be enough
    comptime HitToTuple = OneToManyAssoc[
        Self.tindex_type, -1, Int(4 * Self.maxTuples)
    ]
    comptime TupleMultiplicity = OneToManyAssoc[
        Self.tindex_type, 8, Int(Self.maxTuples)
    ]