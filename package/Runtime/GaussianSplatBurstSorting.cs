// SPDX-License-Identifier: MIT

using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

namespace GaussianSplatting.Runtime
{
    /// <summary>
    /// Burst-compiled sorting jobs for Gaussian splat octree.
    /// Provides high-performance distance-based sorting without GC allocations.
    /// </summary>
    public static class GaussianSplatBurstSorting
    {
        /// <summary>
        /// Metadata for a node's data range in the flattened array.
        /// </summary>
        public struct NodeSortRange
        {
            public int offset;
            public int length;
        }

        /// <summary>
        /// Custom comparer for sorting splat indices by distance from camera.
        /// Used with Unity's NativeSortExtension for optimal performance.
        /// </summary>
        [BurstCompile]
        public struct SplatDistanceComparer : System.Collections.Generic.IComparer<int>
        {
            [ReadOnly] public NativeArray<float3> allPositions;
            public float3 cameraPosition;

            public int Compare(int indexA, int indexB)
            {
                float distA = GetSquaredDistance(indexA);
                float distB = GetSquaredDistance(indexB);
                return distA.CompareTo(distB);
            }

            float GetSquaredDistance(int splatIndex)
            {
                if (splatIndex >= 0 && splatIndex < allPositions.Length)
                {
                    float3 pos = allPositions[splatIndex];
                    float3 diff = pos - cameraPosition;
                    return math.lengthsq(diff);
                }
                return 0f;
            }
        }

        /// <summary>
        /// Burst-compiled parallel job for sorting multiple nodes concurrently.
        /// Uses a flattened array with offset/length metadata to avoid nested containers.
        /// Each job instance sorts one node's splats in parallel using IJobParallelFor.
        /// </summary>
        [BurstCompile]
        public struct SortMultipleNodesJob : IJobParallelFor
        {
            [NativeDisableParallelForRestriction]
            public NativeArray<int> flattenedSplatIndices;
            [ReadOnly] public NativeArray<NodeSortRange> nodeRanges;
            [ReadOnly] public NativeArray<float3> allPositions;
            public float3 cameraPosition;

            public void Execute(int index)
            {
                var range = nodeRanges[index];
                int count = range.length;

                if (count <= 1) return;

                // Get a slice of the flattened array for this range
                var slice = new NativeSlice<int>(flattenedSplatIndices, range.offset, count);

                // Create comparer for distance-based sorting
                var comparer = new SplatDistanceComparer
                {
                    allPositions = allPositions,
                    cameraPosition = cameraPosition
                };

                // Use Unity's optimized sort with custom comparer
                // This uses intro-sort (hybrid quicksort/heapsort/insertion) which is faster
                // and has better cache performance than our manual quicksort
                slice.Sort(comparer);
            }
        }
    }
}
