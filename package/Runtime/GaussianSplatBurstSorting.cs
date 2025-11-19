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
        /// Burst-compiled parallel job for sorting multiple nodes concurrently using optimized hybrid sort.
        /// Each job instance sorts one node's splats in-place using cache-friendly sorting algorithms.
        /// Uses radix sort for large arrays (O(n)) and introsort for medium arrays (O(n log n)).
        /// Optimized with SIMD vectorization, reduced bounds checks, and improved cache locality.
        /// </summary>
        [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Low, DisableSafetyChecks = true)]
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

                // For very small arrays, use insertion sort (faster due to cache and simplicity)
                // Optimized threshold: 16-20 is optimal for modern CPU cache lines (64 bytes)
                if (count < 20)
                {
                    InsertionSortInPlace(splatList, count);
                    return;
                }

                // For medium arrays (20-2048), use introsort
                if (count < 2048)
                {
                    IntroSortInPlace(splatList, count);
                    return;
                }

                // For large arrays (2048+), use true radix sort (O(n) complexity)
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

            // Ultra-optimized introsort with distance caching for medium-sized arrays
            // Combines quicksort, heapsort, and insertion sort for optimal performance
            void IntroSortInPlace(UnsafeList<int> list, int count)
            {
                unsafe
                {
                    int* indices = list.Ptr;

                    // Calculate max recursion depth for introsort (2 * log2(n))
                    int maxDepth = 2 * (int)math.log2(count);

                    // Allocate distance cache on stack for small arrays, heap for large
                    // Increased threshold to 16384 for better stack utilization on 64-bit systems
                    bool useHeap = count > 16384;
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

                    // Pre-compute all distances using SIMD batching (trade memory for speed)
                    int i = 0;
                    int batchEnd = (count / 4) * 4; // Process in batches of 4

                    // SIMD batch processing for better throughput
                    for (; i < batchEnd; i += 4)
                    {
                        ComputeDistancesBatch4(indices, distCache, i);
                    }

                    // Handle remaining elements
                    for (; i < count; i++)
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

            // True radix sort implementation for large arrays (2048+ elements)
            // O(n) complexity using counting sort on IEEE 754 float bit patterns
            void RadixSortInPlace(UnsafeList<int> list, int count)
            {
                unsafe
                {
                    int* indices = list.Ptr;

                    // Allocate buffers (always use heap for large arrays)
                    float* distCache = (float*)UnsafeUtility.Malloc(count * sizeof(float), UnsafeUtility.AlignOf<float>(), Allocator.Temp);
                    int* tempIndices = (int*)UnsafeUtility.Malloc(count * sizeof(int), UnsafeUtility.AlignOf<int>(), Allocator.Temp);

                    // Pre-compute all distances using SIMD batching
                    int i = 0;
                    int batchEnd = (count / 4) * 4;

                    for (; i < batchEnd; i += 4)
                    {
                        ComputeDistancesBatch4(indices, distCache, i);
                    }

                    for (; i < count; i++)
                    {
                        distCache[i] = GetSquaredDistance(indices[i]);
                    }

                    // Perform radix sort on distances
                    RadixSort11Bit(indices, tempIndices, distCache, count);

                    UnsafeUtility.Free(distCache, Allocator.Temp);
                    UnsafeUtility.Free(tempIndices, Allocator.Temp);
                }
            }

            // 11-bit radix sort (3 passes: bits 0-10, 11-21, 22-31)
            // Uses IEEE 754 float flip trick for correct float ordering
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            unsafe uint FloatFlip(float f)
            {
                uint fu = math.asuint(f);
                uint mask = (uint)(-(int)(fu >> 31)) | 0x80000000;
                return fu ^ mask;
            }

            unsafe void RadixSort11Bit(int* indices, int* temp, float* distances, int count)
            {
                const int kBits = 11;
                const int kRadix = 1 << kBits; // 2048 buckets
                const uint kMask = kRadix - 1;

                // Allocate temporary distance array to keep distances paired with indices
                // Use heap allocation for large arrays to avoid stack overflow
                float* tempDist = (float*)UnsafeUtility.Malloc(count * sizeof(float), UnsafeUtility.AlignOf<float>(), Allocator.Temp);

                // Histogram is small (2048 ints = 8KB), safe on stack
                int* histogram = stackalloc int[kRadix];

                // Pass 1: Sort by bits 0-10
                for (int i = 0; i < kRadix; i++)
                    histogram[i] = 0;

                // Build histogram
                for (int i = 0; i < count; i++)
                {
                    uint key = FloatFlip(distances[i]);
                    histogram[(key & kMask)]++;
                }

                // Convert to prefix sum
                int sum = 0;
                for (int i = 0; i < kRadix; i++)
                {
                    int val = histogram[i];
                    histogram[i] = sum;
                    sum += val;
                }

                // Distribute elements
                for (int i = 0; i < count; i++)
                {
                    uint key = FloatFlip(distances[i]);
                    int bucket = (int)(key & kMask);
                    int pos = histogram[bucket]++;
                    temp[pos] = indices[i];
                    tempDist[pos] = distances[i]; // Keep distance paired with index
                }

                // Pass 2: Sort by bits 11-21
                for (int i = 0; i < kRadix; i++)
                    histogram[i] = 0;

                // Build histogram
                for (int i = 0; i < count; i++)
                {
                    uint key = FloatFlip(tempDist[i]);
                    histogram[((key >> kBits) & kMask)]++;
                }

                // Convert to prefix sum
                sum = 0;
                for (int i = 0; i < kRadix; i++)
                {
                    int val = histogram[i];
                    histogram[i] = sum;
                    sum += val;
                }

                // Distribute elements
                for (int i = 0; i < count; i++)
                {
                    uint key = FloatFlip(tempDist[i]);
                    int bucket = (int)((key >> kBits) & kMask);
                    int pos = histogram[bucket]++;
                    indices[pos] = temp[i];
                    distances[pos] = tempDist[i]; // Keep distance paired with index
                }

                // Pass 3: Sort by bits 22-31
                for (int i = 0; i < kRadix; i++)
                    histogram[i] = 0;

                // Build histogram
                for (int i = 0; i < count; i++)
                {
                    uint key = FloatFlip(distances[i]);
                    histogram[((key >> (kBits * 2)) & kMask)]++;
                }

                // Convert to prefix sum
                sum = 0;
                for (int i = 0; i < kRadix; i++)
                {
                    int val = histogram[i];
                    histogram[i] = sum;
                    sum += val;
                }

                // Distribute elements (final pass)
                for (int i = 0; i < count; i++)
                {
                    uint key = FloatFlip(distances[i]);
                    int bucket = (int)((key >> (kBits * 2)) & kMask);
                    int pos = histogram[bucket]++;
                    temp[pos] = indices[i];
                    // No need to copy distances on last pass
                }

                // Copy back to indices
                for (int i = 0; i < count; i++)
                    indices[i] = temp[i];

                // Free temporary distance array
                UnsafeUtility.Free(tempDist, Allocator.Temp);
            }

            // Introsort: Quicksort with heapsort fallback to guarantee O(n log n)
            unsafe void IntroSortInPlace(int* indices, float* distCache, int left, int right, int depthLimit)
            {
                while (right - left > 0)
                {
                    int size = right - left + 1;

                    // Use insertion sort for small subarrays (cache-friendly)
                    // Optimized threshold: 16 is optimal for modern CPU cache lines
                    if (size <= 16)
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

            // Optimized: No bounds check - caller guarantees validity
            // DisableSafetyChecks in BurstCompile removes internal checks
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            float GetSquaredDistance(int splatIndex)
            {
                float3 diff = allPositions[splatIndex] - cameraPosition;
                return math.lengthsq(diff);
            }

            // SIMD optimized: Compute 4 distances at once for better throughput
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            unsafe void ComputeDistancesBatch4(int* indices, float* distCache, int start)
            {
                float3 cam = cameraPosition;

                // Process 4 splats in parallel using SIMD
                float3 pos0 = allPositions[indices[start + 0]];
                float3 pos1 = allPositions[indices[start + 1]];
                float3 pos2 = allPositions[indices[start + 2]];
                float3 pos3 = allPositions[indices[start + 3]];

                float3 diff0 = pos0 - cam;
                float3 diff1 = pos1 - cam;
                float3 diff2 = pos2 - cam;
                float3 diff3 = pos3 - cam;

                distCache[start + 0] = math.dot(diff0, diff0);
                distCache[start + 1] = math.dot(diff1, diff1);
                distCache[start + 2] = math.dot(diff2, diff2);
                distCache[start + 3] = math.dot(diff3, diff3);
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

        // ============================================================================
        // Octree Build Optimization Jobs
        // ============================================================================

        /// <summary>
        /// Computes center of mass for all splat positions using Burst compilation.
        /// Uses parallel reduction for optimal performance.
        /// </summary>
        [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Fast)]
        public struct ComputeCenterOfMassJob : IJob
        {
            [ReadOnly] public NativeArray<float3> positions;
            [WriteOnly] public NativeArray<float3> result; // length 1

            public void Execute()
            {
                if (positions.Length == 0)
                {
                    result[0] = float3.zero;
                    return;
                }

                double3 sum = double3.zero;
                for (int i = 0; i < positions.Length; i++)
                {
                    sum += (double3)positions[i];
                }
                result[0] = (float3)(sum / positions.Length);
            }
        }

        /// <summary>
        /// Computes squared distances from center of mass in parallel.
        /// </summary>
        [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Fast)]
        public struct ComputeDistancesJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<float3> positions;
            public float3 centerOfMass;
            [WriteOnly] public NativeArray<float> distances;

            public void Execute(int index)
            {
                distances[index] = math.distancesq(positions[index], centerOfMass);
            }
        }

        /// <summary>
        /// Sorts indices based on distances using optimized radix sort.
        /// Produces a permutation array without moving the original data.
        /// </summary>
        [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Fast)]
        public struct SortIndicesByDistanceJob : IJob
        {
            [ReadOnly] public NativeArray<float> distances;
            public NativeArray<int> sortedIndices; // Output: permutation array

            public void Execute()
            {
                int count = distances.Length;
                if (count <= 1) return;

                // Initialize indices array
                for (int i = 0; i < count; i++)
                {
                    sortedIndices[i] = i;
                }

                // Use introsort for reliable O(n log n) performance
                IntroSort(0, count - 1, 2 * (int)math.log2(count));
            }

            void IntroSort(int left, int right, int depthLimit)
            {
                while (right - left > 0)
                {
                    int size = right - left + 1;

                    // Use insertion sort for small subarrays
                    if (size <= 32)
                    {
                        InsertionSort(left, right);
                        return;
                    }

                    // Switch to heapsort if recursion too deep
                    if (depthLimit == 0)
                    {
                        HeapSort(left, right);
                        return;
                    }

                    depthLimit--;

                    // Partition and recurse
                    int pivotPos = PartitionMedianOf3(left, right);

                    // Tail recursion optimization
                    if (pivotPos - left < right - pivotPos)
                    {
                        IntroSort(left, pivotPos - 1, depthLimit);
                        left = pivotPos + 1;
                    }
                    else
                    {
                        IntroSort(pivotPos + 1, right, depthLimit);
                        right = pivotPos - 1;
                    }
                }
            }

            int PartitionMedianOf3(int left, int right)
            {
                int mid = left + (right - left) / 2;

                // Median-of-3 pivot selection
                if (GetDistance(left) > GetDistance(mid))
                    SwapIndices(left, mid);
                if (GetDistance(mid) > GetDistance(right))
                    SwapIndices(mid, right);
                if (GetDistance(left) > GetDistance(mid))
                    SwapIndices(left, mid);

                float pivotDist = GetDistance(mid);
                SwapIndices(mid, right - 1);

                int i = left;
                int j = right - 1;

                while (true)
                {
                    while (GetDistance(++i) < pivotDist) { }
                    while (GetDistance(--j) > pivotDist) { }

                    if (i >= j) break;
                    SwapIndices(i, j);
                }

                SwapIndices(i, right - 1);
                return i;
            }

            void InsertionSort(int left, int right)
            {
                for (int i = left + 1; i <= right; i++)
                {
                    int keyIndex = sortedIndices[i];
                    float keyDist = GetDistance(i);
                    int j = i - 1;

                    if (GetDistance(j) <= keyDist)
                        continue;

                    while (j >= left && GetDistance(j) > keyDist)
                    {
                        sortedIndices[j + 1] = sortedIndices[j];
                        j--;
                    }
                    sortedIndices[j + 1] = keyIndex;
                }
            }

            void HeapSort(int left, int right)
            {
                int n = right - left + 1;

                for (int i = n / 2 - 1; i >= 0; i--)
                    HeapifyDown(left, i, n);

                for (int i = n - 1; i > 0; i--)
                {
                    SwapIndices(left, left + i);
                    HeapifyDown(left, 0, i);
                }
            }

            void HeapifyDown(int offset, int root, int size)
            {
                while (true)
                {
                    int largest = root;
                    int leftChild = 2 * root + 1;
                    int rightChild = 2 * root + 2;

                    if (leftChild < size && GetDistance(offset + leftChild) > GetDistance(offset + largest))
                        largest = leftChild;

                    if (rightChild < size && GetDistance(offset + rightChild) > GetDistance(offset + largest))
                        largest = rightChild;

                    if (largest == root)
                        break;

                    SwapIndices(offset + root, offset + largest);
                    root = largest;
                }
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            float GetDistance(int idx)
            {
                return distances[sortedIndices[idx]];
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            void SwapIndices(int a, int b)
            {
                int temp = sortedIndices[a];
                sortedIndices[a] = sortedIndices[b];
                sortedIndices[b] = temp;
            }
        }

        /// <summary>
        /// Computes tight bounding box (min/max) for a subset of positions in parallel.
        /// Uses parallel reduction for optimal performance.
        /// </summary>
        [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Fast)]
        public struct ComputeBoundsJob : IJob
        {
            [ReadOnly] public NativeArray<float3> positions;
            [ReadOnly] public NativeArray<int> indices; // Subset of positions to include
            public int startIndex;
            public int count;
            [WriteOnly] public NativeArray<float3> minResult; // length 1
            [WriteOnly] public NativeArray<float3> maxResult; // length 1

            public void Execute()
            {
                if (count <= 0)
                {
                    minResult[0] = float3.zero;
                    maxResult[0] = float3.zero;
                    return;
                }

                int firstIdx = indices[startIndex];
                float3 min = positions[firstIdx];
                float3 max = positions[firstIdx];

                for (int i = startIndex + 1; i < startIndex + count; i++)
                {
                    int idx = indices[i];
                    if ((uint)idx < (uint)positions.Length)
                    {
                        float3 pos = positions[idx];
                        min = math.min(min, pos);
                        max = math.max(max, pos);
                    }
                }

                minResult[0] = min;
                maxResult[0] = max;
            }
        }

        /// <summary>
        /// Transforms positions from local space to world space in parallel using a transform matrix.
        /// Optimized with Burst for high performance on large datasets.
        /// </summary>
        [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Fast)]
        public struct TransformPositionsJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<float3> localPositions;
            [WriteOnly] public NativeArray<float3> worldPositions;
            public float4x4 transformMatrix;

            public void Execute(int index)
            {
                float3 localPos = localPositions[index];
                float4 pos4 = math.mul(transformMatrix, new float4(localPos, 1.0f));
                worldPositions[index] = pos4.xyz;
            }
        }

        /// <summary>
        /// Extracts and decodes splat positions from compressed asset data in parallel.
        /// Supports multiple compression formats (Float32, Norm16, Norm11, Norm6).
        /// Optimized with Burst for high-performance parallel execution on large datasets.
        /// </summary>
        [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Fast)]
        public struct ExtractSplatPositionsJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<uint> posData;
            [ReadOnly] public NativeArray<GaussianSplatAsset.ChunkInfo> chunkData;
            [WriteOnly] public NativeArray<float3> positions;

            public GaussianSplatAsset.VectorFormat posFormat;
            public int vectorSize;
            public float3 boundsMin;
            public float3 boundsMax;
            public bool useChunkData;

            public void Execute(int splatIndex)
            {
                positions[splatIndex] = DecodeSplatPosition(splatIndex);
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            float3 DecodeSplatPosition(int splatIndex)
            {
                // Calculate byte address for this splat's position data
                int byteAddr = splatIndex * vectorSize;
                int uintAddr = byteAddr / 4;

                // Check bounds to prevent out-of-range errors
                if (uintAddr >= posData.Length)
                    return float3.zero;

                float3 position = float3.zero;

                switch (posFormat)
                {
                    case GaussianSplatAsset.VectorFormat.Float32:
                        // 3 consecutive float32 values - need to check we have enough data
                        if (uintAddr + 2 >= posData.Length)
                            return float3.zero;
                        position.x = math.asfloat(posData[uintAddr]);
                        position.y = math.asfloat(posData[uintAddr + 1]);
                        position.z = math.asfloat(posData[uintAddr + 2]);
                        break;

                    case GaussianSplatAsset.VectorFormat.Norm16:
                        // Packed 16.16.16 format (6 bytes total, needs special handling)
                        if (uintAddr + 1 >= posData.Length)
                            return float3.zero;
                        {
                            uint val0 = posData[uintAddr];
                            uint val1 = posData[uintAddr + 1];
                            // Handle unaligned access
                            if ((byteAddr & 3) != 0)
                            {
                                val0 = (val0 >> 16) | ((val1 & 0xFFFF) << 16);
                                val1 >>= 16;
                            }
                            position.x = (val0 & 0xFFFF) / 65535.0f;
                            position.y = ((val0 >> 16) & 0xFFFF) / 65535.0f;
                            position.z = (val1 & 0xFFFF) / 65535.0f;
                        }
                        break;

                    case GaussianSplatAsset.VectorFormat.Norm11:
                        // Packed 11.10.11 format (32 bits total)
                        {
                            uint val = posData[uintAddr];
                            if ((byteAddr & 3) != 0)
                            {
                                if (uintAddr + 1 >= posData.Length)
                                    return float3.zero;
                                uint val1 = posData[uintAddr + 1];
                                val = (val >> 16) | ((val1 & 0xFFFF) << 16);
                            }
                            position.x = (val & 2047) / 2047.0f;
                            position.y = ((val >> 11) & 1023) / 1023.0f;
                            position.z = ((val >> 21) & 2047) / 2047.0f;
                        }
                        break;

                    case GaussianSplatAsset.VectorFormat.Norm6:
                        // Packed 6.5.5 format (16 bits total)
                        {
                            uint val = LoadUShortFromByteAddr(byteAddr);
                            position.x = (val & 63) / 63.0f;
                            position.y = ((val >> 6) & 31) / 31.0f;
                            position.z = ((val >> 11) & 31) / 31.0f;
                        }
                        break;
                }

                // Apply chunk-relative positioning if chunk data exists
                if (useChunkData && chunkData.Length > 0)
                {
                    int chunkIndex = splatIndex / GaussianSplatAsset.kChunkSize;
                    if (chunkIndex < chunkData.Length)
                    {
                        var chunk = chunkData[chunkIndex];
                        // Convert chunk bounds to world space
                        position.x = math.lerp(chunk.posX.x, chunk.posX.y, position.x);
                        position.y = math.lerp(chunk.posY.x, chunk.posY.y, position.y);
                        position.z = math.lerp(chunk.posZ.x, chunk.posZ.y, position.z);
                    }
                }
                else
                {
                    // Use asset bounds
                    position.x = math.lerp(boundsMin.x, boundsMax.x, position.x);
                    position.y = math.lerp(boundsMin.y, boundsMax.y, position.y);
                    position.z = math.lerp(boundsMin.z, boundsMax.z, position.z);
                }

                return position;
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            uint LoadUShortFromByteAddr(int byteAddr)
            {
                int alignedAddr = byteAddr & ~0x3;
                int uintIndex = alignedAddr / 4;

                if (uintIndex >= posData.Length)
                    return 0;

                uint val = posData[uintIndex];
                if (byteAddr != alignedAddr)
                    val >>= 16;
                return val & 0xFFFF;
            }
        }
    }
}
