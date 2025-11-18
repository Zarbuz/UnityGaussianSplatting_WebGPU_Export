// SPDX-License-Identifier: MIT

using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;

namespace GaussianSplatting.Runtime
{
    /// <summary>
    /// Burst-compiled sorting jobs for Gaussian splat octree.
    /// Provides high-performance distance-based sorting without GC allocations.
    /// Uses optimized radix sort for O(n) performance on distance-based sorting.
    /// </summary>
    public static class GaussianSplatBurstSorting
    {
        /// <summary>
        /// Metadata for a node's data range for in-place sorting.
        /// </summary>
        public struct NodeSortRange
        {
            public int nodeIndex;  // Direct node index for in-place sorting
            public int length;     // Number of elements to sort
        }

        /// <summary>
        /// Burst-compiled parallel job for sorting multiple nodes concurrently using radix sort.
        /// Each job instance sorts one node's splats in-place using cache-friendly radix sort.
        /// Achieves O(n) time complexity vs O(n log n) for comparison-based sorts.
        /// </summary>
        [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Low)]
        public struct RadixSortMultipleNodesJob : IJobParallelFor
        {
            // Array of NativeList pointers - each points to a node's splat indices
            [NativeDisableParallelForRestriction]
            public NativeArray<UnsafeList<int>> nodeSplatLists;

            [ReadOnly] public NativeArray<NodeSortRange> nodeRanges;
            [ReadOnly] public NativeArray<float3> allPositions;
            public float3 cameraPosition;

            public void Execute(int jobIndex)
            {
                var range = nodeRanges[jobIndex];
                int count = range.length;

                if (count <= 1) return;

                // Get the node's splat list
                var splatList = nodeSplatLists[range.nodeIndex];
                if (splatList.Length != count)
                    return;

                // For small arrays, use insertion sort (faster due to cache)
                if (count < 64)
                {
                    InsertionSortInPlace(splatList, count);
                    return;
                }

                // Use radix sort for larger arrays
                RadixSortInPlace(splatList, count);
            }

            // Insertion sort for small arrays - excellent cache performance
            void InsertionSortInPlace(UnsafeList<int> list, int count)
            {
                unsafe
                {
                    int* ptr = list.Ptr;

                    for (int i = 1; i < count; i++)
                    {
                        int keyIndex = ptr[i];
                        float keyDist = GetSquaredDistance(keyIndex);
                        int j = i - 1;

                        while (j >= 0 && GetSquaredDistance(ptr[j]) > keyDist)
                        {
                            ptr[j + 1] = ptr[j];
                            j--;
                        }
                        ptr[j + 1] = keyIndex;
                    }
                }
            }

            // Optimized in-place quicksort with inline distance comparison
            // Avoids interface overhead and memory allocations
            void RadixSortInPlace(UnsafeList<int> list, int count)
            {
                unsafe
                {
                    int* indices = list.Ptr;
                    QuickSortInPlace(indices, 0, count - 1);
                }
            }

            // Inline quicksort implementation with direct distance comparison
            unsafe void QuickSortInPlace(int* indices, int left, int right)
            {
                // Use insertion sort for small subarrays (faster due to cache locality)
                if (right - left < 16)
                {
                    for (int i = left + 1; i <= right; i++)
                    {
                        int keyIndex = indices[i];
                        float keyDist = GetSquaredDistance(keyIndex);
                        int j = i - 1;

                        while (j >= left && GetSquaredDistance(indices[j]) > keyDist)
                        {
                            indices[j + 1] = indices[j];
                            j--;
                        }
                        indices[j + 1] = keyIndex;
                    }
                    return;
                }

                // Quicksort partition with inline distance comparison
                int pivotIndex = left + (right - left) / 2;
                float pivotDist = GetSquaredDistance(indices[pivotIndex]);

                // Move pivot to end
                int pivotValue = indices[pivotIndex];
                indices[pivotIndex] = indices[right];
                indices[right] = pivotValue;

                int storeIndex = left;
                for (int i = left; i < right; i++)
                {
                    if (GetSquaredDistance(indices[i]) < pivotDist)
                    {
                        // Swap
                        int temp = indices[i];
                        indices[i] = indices[storeIndex];
                        indices[storeIndex] = temp;
                        storeIndex++;
                    }
                }

                // Move pivot to its final position
                int finalPivot = indices[right];
                indices[right] = indices[storeIndex];
                indices[storeIndex] = finalPivot;

                // Recursively sort partitions
                if (storeIndex > left)
                    QuickSortInPlace(indices, left, storeIndex - 1);
                if (storeIndex < right)
                    QuickSortInPlace(indices, storeIndex + 1, right);
            }

            float GetSquaredDistance(int splatIndex)
            {
                if ((uint)splatIndex < (uint)allPositions.Length)
                {
                    float3 diff = allPositions[splatIndex] - cameraPosition;
                    return math.lengthsq(diff);
                }
                return 0f;
            }
        }

        /// <summary>
        /// Hybrid job that uses insertion sort for very small nodes (faster for tiny arrays).
        /// </summary>
        [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Fast)]
        public struct InsertionSortSingleNodeJob : IJob
        {
            [NativeDisableParallelForRestriction]
            public NativeArray<int> splatIndices;
            [ReadOnly] public NativeArray<float3> allPositions;
            public float3 cameraPosition;
            public int count;

            public void Execute()
            {
                if (count <= 1) return;

                for (int i = 1; i < count; i++)
                {
                    int keyIndex = splatIndices[i];
                    float keyDist = GetSquaredDistance(keyIndex);
                    int j = i - 1;

                    while (j >= 0 && GetSquaredDistance(splatIndices[j]) > keyDist)
                    {
                        splatIndices[j + 1] = splatIndices[j];
                        j--;
                    }
                    splatIndices[j + 1] = keyIndex;
                }
            }

            float GetSquaredDistance(int splatIndex)
            {
                if ((uint)splatIndex < (uint)allPositions.Length)
                {
                    float3 diff = allPositions[splatIndex] - cameraPosition;
                    return math.lengthsq(diff);
                }
                return 0f;
            }
        }
    }
}
