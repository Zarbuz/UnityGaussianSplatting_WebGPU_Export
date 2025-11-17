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
        /// Burst-compiled job for sorting splats within a single node by distance from camera.
        /// Uses hybrid sorting: insertion sort for small arrays, quicksort for larger ones.
        /// </summary>
        [BurstCompile]
        public struct SortNodeSplatsJob : IJob
        {
            public NativeArray<int> splatIndices;
            [ReadOnly] public NativeArray<float3> allPositions;
            public float3 cameraPosition;

            public void Execute()
            {
                int count = splatIndices.Length;
                if (count <= 1) return;

                // Use insertion sort for small arrays (better cache locality)
                // Use quicksort for larger arrays
                if (count < 32)
                {
                    InsertionSort();
                }
                else
                {
                    QuickSort(0, count - 1);
                }
            }

            void InsertionSort()
            {
                int count = splatIndices.Length;
                for (int i = 1; i < count; i++)
                {
                    int key = splatIndices[i];
                    float keyDist = GetSquaredDistance(key);
                    int j = i - 1;

                    while (j >= 0 && GetSquaredDistance(splatIndices[j]) > keyDist)
                    {
                        splatIndices[j + 1] = splatIndices[j];
                        j--;
                    }
                    splatIndices[j + 1] = key;
                }
            }

            void QuickSort(int low, int high)
            {
                // Iterative quicksort using a stack to avoid recursion depth issues
                const int stackSize = 128; // Increased for larger arrays (log2(max array size) * 2)
                int stackPtr = 0;
                var stack = new NativeArray<int>(stackSize, Allocator.Temp);

                stack[stackPtr++] = low;
                stack[stackPtr++] = high;

                while (stackPtr > 0)
                {
                    high = stack[--stackPtr];
                    low = stack[--stackPtr];

                    if (low < high)
                    {
                        int pi = Partition(low, high);

                        // Push smaller partition first to limit stack depth
                        if (pi - low < high - pi)
                        {
                            // Right partition is larger, push it first (will be processed last)
                            if (pi + 1 < high && stackPtr + 2 <= stackSize)
                            {
                                stack[stackPtr++] = pi + 1;
                                stack[stackPtr++] = high;
                            }
                            // Left partition is smaller, push it second (will be processed first)
                            if (pi - 1 > low && stackPtr + 2 <= stackSize)
                            {
                                stack[stackPtr++] = low;
                                stack[stackPtr++] = pi - 1;
                            }
                        }
                        else
                        {
                            // Left partition is larger, push it first (will be processed last)
                            if (pi - 1 > low && stackPtr + 2 <= stackSize)
                            {
                                stack[stackPtr++] = low;
                                stack[stackPtr++] = pi - 1;
                            }
                            // Right partition is smaller, push it second (will be processed first)
                            if (pi + 1 < high && stackPtr + 2 <= stackSize)
                            {
                                stack[stackPtr++] = pi + 1;
                                stack[stackPtr++] = high;
                            }
                        }
                    }
                }

                stack.Dispose();
            }

            int Partition(int low, int high)
            {
                int pivot = splatIndices[high];
                float pivotDist = GetSquaredDistance(pivot);
                int i = low - 1;

                for (int j = low; j < high; j++)
                {
                    if (GetSquaredDistance(splatIndices[j]) <= pivotDist)
                    {
                        i++;
                        // Swap
                        int temp = splatIndices[i];
                        splatIndices[i] = splatIndices[j];
                        splatIndices[j] = temp;
                    }
                }
                // Swap pivot
                int temp2 = splatIndices[i + 1];
                splatIndices[i + 1] = splatIndices[high];
                splatIndices[high] = temp2;
                return i + 1;
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
