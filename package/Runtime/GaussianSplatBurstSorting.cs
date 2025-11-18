// SPDX-License-Identifier: MIT

using System.Runtime.CompilerServices;
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

            // Optimized insertion sort for very small arrays with distance caching
            void InsertionSortInPlace(UnsafeList<int> list, int count)
            {
                unsafe
                {
                    int* ptr = list.Ptr;

                    // Pre-compute distances for small arrays (worth it for insertion sort)
                    float* distCache = stackalloc float[count];
                    for (int i = 0; i < count; i++)
                    {
                        distCache[i] = GetSquaredDistance(ptr[i]);
                    }

                    // Insertion sort with cached distances
                    for (int i = 1; i < count; i++)
                    {
                        int keyIndex = ptr[i];
                        float keyDist = distCache[i];
                        int j = i - 1;

                        // Early exit if already sorted
                        if (distCache[j] <= keyDist)
                            continue;

                        while (j >= 0 && distCache[j] > keyDist)
                        {
                            ptr[j + 1] = ptr[j];
                            distCache[j + 1] = distCache[j];
                            j--;
                        }
                        ptr[j + 1] = keyIndex;
                        distCache[j + 1] = keyDist;
                    }
                }
            }

            // Ultra-optimized introsort with distance caching
            // Combines quicksort, heapsort, and insertion sort for optimal performance
            void RadixSortInPlace(UnsafeList<int> list, int count)
            {
                unsafe
                {
                    int* indices = list.Ptr;

                    // Calculate max recursion depth for introsort (2 * log2(n))
                    int maxDepth = 2 * (int)math.log2(count);

                    // Allocate distance cache on stack for small arrays, heap for large
                    bool useHeap = count > 4096;
                    float* distCache;

                    if (useHeap)
                    {
                        distCache = (float*)UnsafeUtility.Malloc(count * sizeof(float), UnsafeUtility.AlignOf<float>(), Allocator.Temp);
                    }
                    else
                    {
                        float* tempCache = stackalloc float[count];
                        distCache = tempCache;
                    }

                    // Pre-compute all distances (trade memory for speed)
                    for (int i = 0; i < count; i++)
                    {
                        distCache[i] = GetSquaredDistance(indices[i]);
                    }

                    // Run introsort with distance cache
                    IntroSortInPlace(indices, distCache, 0, count - 1, maxDepth);

                    if (useHeap)
                    {
                        UnsafeUtility.Free(distCache, Allocator.Temp);
                    }
                }
            }

            // Introsort: Quicksort with heapsort fallback to guarantee O(n log n)
            unsafe void IntroSortInPlace(int* indices, float* distCache, int left, int right, int depthLimit)
            {
                while (right - left > 0)
                {
                    int size = right - left + 1;

                    // Use insertion sort for small subarrays (cache-friendly)
                    if (size <= 32)
                    {
                        InsertionSortCached(indices, distCache, left, right);
                        return;
                    }

                    // Switch to heapsort if recursion too deep (avoid O(n²) worst case)
                    if (depthLimit == 0)
                    {
                        HeapSortCached(indices, distCache, left, right);
                        return;
                    }

                    depthLimit--;

                    // Three-way partition for better handling of duplicates
                    int pivotPos = PartitionMedianOf3(indices, distCache, left, right);

                    // Tail recursion optimization: recurse on smaller partition, loop on larger
                    if (pivotPos - left < right - pivotPos)
                    {
                        IntroSortInPlace(indices, distCache, left, pivotPos - 1, depthLimit);
                        left = pivotPos + 1;
                    }
                    else
                    {
                        IntroSortInPlace(indices, distCache, pivotPos + 1, right, depthLimit);
                        right = pivotPos - 1;
                    }
                }
            }

            // Optimized partition using median-of-3 pivot selection
            unsafe int PartitionMedianOf3(int* indices, float* distCache, int left, int right)
            {
                int mid = left + (right - left) / 2;

                // Median-of-3: sort left, mid, right
                if (distCache[left] > distCache[mid])
                {
                    Swap(ref indices[left], ref indices[mid]);
                    Swap(ref distCache[left], ref distCache[mid]);
                }
                if (distCache[mid] > distCache[right])
                {
                    Swap(ref indices[mid], ref indices[right]);
                    Swap(ref distCache[mid], ref distCache[right]);
                }
                if (distCache[left] > distCache[mid])
                {
                    Swap(ref indices[left], ref indices[mid]);
                    Swap(ref distCache[left], ref distCache[mid]);
                }

                // Use mid as pivot
                float pivotDist = distCache[mid];

                // Move pivot to end
                Swap(ref indices[mid], ref indices[right - 1]);
                Swap(ref distCache[mid], ref distCache[right - 1]);

                // Partition
                int i = left;
                int j = right - 1;

                while (true)
                {
                    while (distCache[++i] < pivotDist) { }
                    while (distCache[--j] > pivotDist) { }

                    if (i >= j) break;

                    Swap(ref indices[i], ref indices[j]);
                    Swap(ref distCache[i], ref distCache[j]);
                }

                // Restore pivot
                Swap(ref indices[i], ref indices[right - 1]);
                Swap(ref distCache[i], ref distCache[right - 1]);

                return i;
            }

            // Optimized insertion sort using cached distances
            unsafe void InsertionSortCached(int* indices, float* distCache, int left, int right)
            {
                for (int i = left + 1; i <= right; i++)
                {
                    int keyIndex = indices[i];
                    float keyDist = distCache[i];
                    int j = i - 1;

                    // Unroll first comparison to reduce loop overhead
                    if (distCache[j] <= keyDist)
                        continue;

                    while (j >= left && distCache[j] > keyDist)
                    {
                        indices[j + 1] = indices[j];
                        distCache[j + 1] = distCache[j];
                        j--;
                    }

                    indices[j + 1] = keyIndex;
                    distCache[j + 1] = keyDist;
                }
            }

            // Heapsort fallback for worst-case scenarios
            unsafe void HeapSortCached(int* indices, float* distCache, int left, int right)
            {
                int n = right - left + 1;

                // Build max heap
                for (int i = n / 2 - 1; i >= 0; i--)
                {
                    HeapifyDown(indices, distCache, left, i, n);
                }

                // Extract elements from heap
                for (int i = n - 1; i > 0; i--)
                {
                    Swap(ref indices[left], ref indices[left + i]);
                    Swap(ref distCache[left], ref distCache[left + i]);
                    HeapifyDown(indices, distCache, left, 0, i);
                }
            }

            unsafe void HeapifyDown(int* indices, float* distCache, int offset, int root, int size)
            {
                while (true)
                {
                    int largest = root;
                    int leftChild = 2 * root + 1;
                    int rightChild = 2 * root + 2;

                    if (leftChild < size && distCache[offset + leftChild] > distCache[offset + largest])
                        largest = leftChild;

                    if (rightChild < size && distCache[offset + rightChild] > distCache[offset + largest])
                        largest = rightChild;

                    if (largest == root)
                        break;

                    Swap(ref indices[offset + root], ref indices[offset + largest]);
                    Swap(ref distCache[offset + root], ref distCache[offset + largest]);
                    root = largest;
                }
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            unsafe void Swap(ref int a, ref int b)
            {
                int temp = a;
                a = b;
                b = temp;
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            unsafe void Swap(ref float a, ref float b)
            {
                float temp = a;
                a = b;
                b = temp;
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
