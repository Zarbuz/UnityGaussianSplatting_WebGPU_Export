// SPDX-License-Identifier: MIT

using System; // Added for Exception
using System.Collections.Generic;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using Unity.Jobs;
using Unity.Burst;
using UnityEngine;
// Added for simple threading support
using System.Threading;
using System.Buffers;
using System.Threading.Tasks;

namespace GaussianSplatting.Runtime
{
    /// <summary>
    /// Octree-based spatial acceleration structure for Gaussian splat frustum culling.
    /// Divides scene bounds into hierarchical octants for efficient culling of static splats.
    /// </summary>
    public class GaussianSplatOctree
    {
        public class OctreeNode
        {
            public Bounds bounds;
            public Vector3 center;
            // For leaf nodes we store original splat indices that lie within this node's bounds.
            public NativeList<int> splatIndices;
            // Child node indices (indices into m_Nodes). Empty for leaf nodes.
            public NativeList<int> childIndices;
            public bool isLeaf;
            // Track if this node's splats are sorted for current camera view
            public bool isSorted;
            // Store the camera position used for last sort (to detect when re-sort needed)
            public Vector3 lastSortCameraPosition;
            // Cached maximum extent (largest half-size axis) for angular sort threshold calculations
            public float maxExtent;

            public void Dispose()
            {
                if (splatIndices.IsCreated)
                    splatIndices.Dispose();
                if (childIndices.IsCreated)
                    childIndices.Dispose();
            }
        }

        public struct SplatInfo
        {
            public float3 position;
            public int originalIndex;
        }

        readonly List<OctreeNode> m_Nodes = new();
        NativeArray<int> m_VisibleSplatIndices;
        bool m_VisibleSplatIndicesValid;
        int m_TotalSplats; // Persist total splat count after releasing build-time list

        // Configuration
        int m_MaxDepth;
        int m_MaxSplatsPerLeaf;
        Bounds m_RootBounds;
        bool m_Built;
        bool m_IsBuilding; // Track if a build is currently in progress

        // GPU buffer for visible splat indices (updated per frame/N frames)
        GraphicsBuffer m_VisibleIndicesBuffer;

        // Outlier splat indices that lie outside the main root bounds (always included in culling)
        NativeList<int> m_OthersIndices;
        bool m_OthersIndicesValid;
        
        // Reusable array for distance sorting to avoid allocations
        (float distance, int index)[] m_DistanceSortArray;

        // Structure to store visible node references with their distance for hierarchical sorting
        struct VisibleNodeRef
        {
            public float distance;
            public int nodeIndex; // Index into m_Nodes instead of copying splat indices
        }

        // Comparer for sorting VisibleNodeRef by distance (front-to-back)
        struct VisibleNodeRefDistanceComparer : System.Collections.Generic.IComparer<VisibleNodeRef>
        {
            public int Compare(VisibleNodeRef a, VisibleNodeRef b)
            {
                return a.distance.CompareTo(b.distance);
            }
        }

        // Reusable NativeList for visible node references during sorting
        NativeList<VisibleNodeRef> m_VisibleNodeRefs;
        bool m_VisibleNodeRefsValid;
        
        // Reusable stack for non-recursive octree traversal
        readonly Stack<int> m_TraversalStack = new();

        // Reusable frustum planes array to avoid GC allocations
        readonly Plane[] m_FrustumPlanes = new Plane[6];
        
        public int maxSortNodesPerFrame = 256; // In sequential path we do sort over time
        // Angular threshold for re-sorting: minimum cosine of angle change before re-sort is needed
        // cosine(15°) ≈ 0.966, cosine(30°) ≈ 0.866, cosine(45°) ≈ 0.707
        public float sortDirectionThreshold = 0.9f; // ~25.8° angle change threshold
        // Simplified outlier sorting strategy:
        // We keep an average radial distance (ring radius) of outliers from the root center.
        // Re-sorting occurs only when camera has moved more than (outlierRingRadius * outlierResortMoveFraction).
        // If the computed ring radius is zero (edge case), we fall back to a small constant.
        public float outlierResortMoveFraction = 0.1f; // 10% of ring radius movement triggers re-sort
        public float minOutlierResortDistance = 0.05f; // Fallback minimum distance if radius very small
        bool m_OthersSorted;            // Track if outliers are currently sorted
        Vector3 m_LastOthersSortCamPos; // Camera position at last outlier sort
        float m_OutlierRingRadius;      // Average radial distance of outliers from scene center

        JobHandle m_SortJobHandle;
        bool m_SortJobRunning;

        // Storage for async job data (must persist until job completes)
        NativeArray<UnsafeList<int>> m_NodeSplatListPointers;
        NativeArray<GaussianSplatBurstSorting.NodeSortRange> m_NodeRanges;
        NativeList<int> m_NodesToSort;
        bool m_SortOutliers;
		Vector3 m_SortJobCameraPosition;

        // Cache to avoid redundant GPU uploads
        int m_LastUploadedSplatCount;
        bool m_BufferNeedsUpload;

        // Cache previous frame data to skip redundant work
        Vector3 m_LastCullCameraPosition;
        Quaternion m_LastCullCameraRotation;
        float m_CullUpdateThreshold = 0.01f; // Skip culling if camera moved less than this

        // Pooled job arrays to avoid per-frame allocations (optimization 4)
        NativeArray<UnsafeList<int>> m_CachedNodeSplatListPointers;
        NativeArray<GaussianSplatBurstSorting.NodeSortRange> m_CachedNodeRanges;
        bool m_JobArraysValid;

        // Global native positions buffer (all splat positions) to avoid per-job copying
        NativeArray<float3> m_AllPositionsNative;
        bool m_AllPositionsNativeValid;

        public int nodeCount => m_Nodes.Count;
        public int totalSplats => m_TotalSplats;
        public bool isBuilt => m_Built;
        public GraphicsBuffer visibleIndicesBuffer => m_VisibleIndicesBuffer;
        public int visibleSplatCount { get; private set; }

        // Helper to get splat position directly from global native buffer
        bool TryGetSplatPosition(int originalIndex, out float3 pos)
        {
            if (m_AllPositionsNativeValid && originalIndex >= 0 && originalIndex < m_AllPositionsNative.Length)
            {
                pos = m_AllPositionsNative[originalIndex];
                return true;
            }
            pos = default;
            return false;
        }

        // Helper to ensure the visible splat indices native array is large enough
        void EnsureVisibleSplatIndicesCapacity(int requiredCapacity)
        {
            if (!m_VisibleSplatIndicesValid || !m_VisibleSplatIndices.IsCreated || m_VisibleSplatIndices.Length < requiredCapacity)
            {
                if (m_VisibleSplatIndicesValid && m_VisibleSplatIndices.IsCreated)
                {
                    try { m_VisibleSplatIndices.Dispose(); } catch {}
                }

                // Allocate with some extra space to avoid frequent reallocations
                int bufferSize = Mathf.NextPowerOfTwo(Mathf.Max(requiredCapacity, 1));
                try
                {
                    m_VisibleSplatIndices = new NativeArray<int>(bufferSize, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
                    m_VisibleSplatIndicesValid = true;
                }
                catch (Exception ex)
                {
                    Debug.LogError($"Failed to allocate visible splat indices native array: {ex.Message}");
                    m_VisibleSplatIndicesValid = false;
                }
            }
        }

        // Helper to ensure pooled job arrays are large enough (optimization 4)
        void EnsureJobArraysCapacity(int requiredCapacity)
        {
            if (!m_JobArraysValid || !m_CachedNodeSplatListPointers.IsCreated || m_CachedNodeSplatListPointers.Length < requiredCapacity)
            {
                if (m_JobArraysValid)
                {
                    if (m_CachedNodeSplatListPointers.IsCreated)
                    {
                        try { m_CachedNodeSplatListPointers.Dispose(); } catch {}
                    }
                    if (m_CachedNodeRanges.IsCreated)
                    {
                        try { m_CachedNodeRanges.Dispose(); } catch {}
                    }
                }

                // Allocate with power-of-2 size to reduce reallocations
                int bufferSize = Mathf.NextPowerOfTwo(Mathf.Max(requiredCapacity, 64));
                try
                {
                    m_CachedNodeSplatListPointers = new NativeArray<UnsafeList<int>>(bufferSize, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
                    m_CachedNodeRanges = new NativeArray<GaussianSplatBurstSorting.NodeSortRange>(bufferSize, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
                    m_JobArraysValid = true;
                }
                catch (Exception ex)
                {
                    Debug.LogError($"Failed to allocate pooled job arrays: {ex.Message}");
                    m_JobArraysValid = false;
                }
            }
        }

        /// <summary>
        /// Initialize octree parameters. Call this before building.
        /// </summary>
        /// <param name="maxDepth">Maximum tree depth (typically 4-6)</param>
        /// <param name="maxSplatsPerLeaf">Maximum splats per leaf node (typically 64-256)</param>
        public void Initialize(int maxDepth = 5, int maxSplatsPerLeaf = 128)
        {
            m_MaxDepth = maxDepth;
            m_MaxSplatsPerLeaf = maxSplatsPerLeaf;

            // Print available system cores (both .NET and Unity reports)
            int envCores = Environment.ProcessorCount;
            int unityCores = SystemInfo.processorCount;
            Debug.Log($"Available cores - Environment.ProcessorCount: {envCores}, SystemInfo.processorCount: {unityCores}");

            // Initialize NativeList for visible node references
            if (!m_VisibleNodeRefsValid || !m_VisibleNodeRefs.IsCreated)
            {
                if (m_VisibleNodeRefsValid && m_VisibleNodeRefs.IsCreated)
                {
                    m_VisibleNodeRefs.Dispose();
                }
                m_VisibleNodeRefs = new NativeList<VisibleNodeRef>(256, Allocator.Persistent);
                m_VisibleNodeRefsValid = true;
            }

            // Initialize NativeList for outlier indices
            if (!m_OthersIndicesValid || !m_OthersIndices.IsCreated)
            {
                if (m_OthersIndicesValid && m_OthersIndices.IsCreated)
                {
                    m_OthersIndices.Dispose();
                }
                m_OthersIndices = new NativeList<int>(256, Allocator.Persistent);
                m_OthersIndicesValid = true;
            }
        }

        /// <summary>
        /// Build octree from splat position data and bounds (synchronous version).
        /// For legacy compatibility. Consider using BuildAsync for better memory management.
        /// NOTE: Octree takes ownership of the splatPositions array - caller should not dispose it.
        /// </summary>
        public void Build(NativeArray<float3> splatPositions, Bounds sceneBounds, float splatPercent)
        {
            // Run the async version synchronously
            var task = BuildAsync(splatPositions, sceneBounds, splatPercent);
            task.Wait();
        }

        /// <summary>
        /// Build octree asynchronously from splat position data and bounds.
        /// Optimized with Burst-compiled jobs and incremental memory cleanup to reduce peak memory usage.
        /// Critical for WebGL builds where WASM heap size is limited.
        /// NOTE: Octree takes ownership of the splatPositions array - caller should not dispose it.
        /// </summary>
        /// <param name="splatPositions">Splat position data (must be Allocator.Persistent, octree takes ownership)</param>
        /// <param name="sceneBounds">Scene bounds</param>
        /// <param name="splatPercent">Percentage of splats to include in octree</param>
        public async Task BuildAsync(NativeArray<float3> splatPositions, Bounds sceneBounds, float splatPercent)
        {
            // Prevent concurrent builds that would dispose nodes while previous build is still running
            if (m_IsBuilding)
            {
                Debug.LogWarning("GaussianSplatOctree.BuildAsync: Build already in progress, ignoring new build request");
                return;
            }

            m_IsBuilding = true;

            try
            {
                Clear();

                if (splatPositions.Length == 0)
                {
                    Debug.LogWarning("GaussianSplatOctree.Build: No splat positions provided");
                    return;
                }

                Debug.Log($"Building octree with {splatPositions.Length} splats, bounds: {sceneBounds}");

                int total = splatPositions.Length;
                m_TotalSplats = total;

                if (!splatPositions.IsCreated)
                {
                    Debug.LogError("Input splatPositions array is not created");
                    return;
                }

            // ============================================================================
            // PHASE 1: Parallel center of mass calculation using Burst
            // ============================================================================
            NativeArray<float3> comResult = default;
            float3 com;

            try
            {
                comResult = new NativeArray<float3>(1, Allocator.TempJob);
                var comJob = new GaussianSplatBurstSorting.ComputeCenterOfMassJob
                {
                    positions = splatPositions, // Use input directly (read-only)
                    result = comResult
                };
                var comHandle = comJob.Schedule();

                // Wait async for COM job
                while (!comHandle.IsCompleted)
                    await Task.Yield();

                comHandle.Complete();
                com = comResult[0];
            }
            finally
            {
                // Clean up Phase 1 allocations immediately
                if (comResult.IsCreated) comResult.Dispose();
            }

            // ============================================================================
            // PHASE 2: Parallel distance calculation using Burst
            // ============================================================================
            NativeArray<float> distances = default;

            try
            {
                // Use Persistent allocator because distances is used after async yields in Phase 3
                distances = new NativeArray<float>(total, Allocator.Persistent);
                var distJob = new GaussianSplatBurstSorting.ComputeDistancesJob
                {
                    positions = splatPositions, // Use input directly (read-only)
                    centerOfMass = com,
                    distances = distances
                };
                var distHandle = distJob.Schedule(total, 512);

                // Wait async for distance job
                while (!distHandle.IsCompleted)
                    await Task.Yield();

                distHandle.Complete();
            }
            catch
            {
                // Cleanup on error
                if (distances.IsCreated) distances.Dispose();
                throw;
            }

            // ============================================================================
            // PHASE 3: Sort indices by distance using Burst
            // ============================================================================
            NativeArray<int> sortedIndices = default;

            try
            {
                // Use Persistent allocator because sortedIndices is used throughout async recursive build
                // TempJob would be invalidated after async yields
                sortedIndices = new NativeArray<int>(total, Allocator.Persistent);
                var sortJob = new GaussianSplatBurstSorting.SortIndicesByDistanceJob
                {
                    distances = distances,
                    sortedIndices = sortedIndices
                };
                var sortHandle = sortJob.Schedule();

                // Wait async for sort job
                while (!sortHandle.IsCompleted)
                    await Task.Yield();

                sortHandle.Complete();
            }
            finally
            {
                // Clean up distances array after sorting
                if (distances.IsCreated) distances.Dispose();
            }

            // ============================================================================
            // PHASE 4: Partition into in-bounds and outliers
            // ============================================================================
            int inCount = Mathf.CeilToInt(total * splatPercent);
            inCount = Mathf.Clamp(inCount, 1, total);
            int othersCount = total - inCount;

            // ============================================================================
            // PHASE 5: Take ownership of positions buffer (no copy needed!)
            // ============================================================================
            if (m_AllPositionsNativeValid)
            {
                if (m_AllPositionsNative.IsCreated) m_AllPositionsNative.Dispose();
                m_AllPositionsNativeValid = false;
            }

            // Take ownership of the input array - saves memory by avoiding a copy!
            m_AllPositionsNative = splatPositions;
            m_AllPositionsNativeValid = true;

            // Yield to allow GC to run
            await Task.Yield();

            // ============================================================================
            // PHASE 6: Compute root bounds for in-bounds splats
            // Use m_AllPositionsNative since splatPositions may be deallocated after yield
            // ============================================================================
            Bounds rootBounds;
            if (inCount > 0 && m_AllPositionsNativeValid)
            {
                NativeArray<float3> minResult = default;
                NativeArray<float3> maxResult = default;

                try
                {
                    minResult = new NativeArray<float3>(1, Allocator.TempJob);
                    maxResult = new NativeArray<float3>(1, Allocator.TempJob);

                    var boundsJob = new GaussianSplatBurstSorting.ComputeBoundsJob
                    {
                        positions = m_AllPositionsNative, // Use persistent buffer
                        indices = sortedIndices,
                        startIndex = 0,
                        count = inCount,
                        minResult = minResult,
                        maxResult = maxResult
                    };
                    boundsJob.Run();

                    float3 min = minResult[0];
                    float3 max = maxResult[0];
                    rootBounds = new Bounds((max + min) * 0.5f, max - min);
                }
                finally
                {
                    if (minResult.IsCreated) minResult.Dispose();
                    if (maxResult.IsCreated) maxResult.Dispose();
                }
            }
            else
            {
                rootBounds = sceneBounds;
            }

            m_RootBounds = rootBounds;

            // ============================================================================
            // PHASE 7: Build octree recursively using sorted indices
            // Use m_AllPositionsNative since splatPositions may be deallocated after yield
            // ============================================================================
            m_Nodes.Clear();

            var rootNode = new OctreeNode
            {
                bounds = m_RootBounds,
                center = m_RootBounds.center,
                splatIndices = new NativeList<int>(Allocator.Persistent),
                childIndices = new NativeList<int>(8, Allocator.Persistent),
                isLeaf = false,
                maxExtent = Mathf.Max(m_RootBounds.extents.x, Mathf.Max(m_RootBounds.extents.y, m_RootBounds.extents.z))
            };
            m_Nodes.Add(rootNode);

            // Build recursively using sorted indices and persistent positions buffer
            // Use async version to allow yielding during build for better memory management
            await BuildRecursiveOptimizedAsync(0, 0, sortedIndices, 0, inCount, m_AllPositionsNative);

            // ============================================================================
            // PHASE 8: Handle outliers
            // ============================================================================
            if (!m_OthersIndicesValid || !m_OthersIndices.IsCreated)
            {
                if (m_OthersIndicesValid && m_OthersIndices.IsCreated)
                    m_OthersIndices.Dispose();
                m_OthersIndices = new NativeList<int>(Allocator.Persistent);
                m_OthersIndicesValid = true;
            }
            else
            {
                m_OthersIndices.Clear();
            }

            if (othersCount > 0)
            {
                for (int i = 0; i < othersCount; i++)
                {
                    int originalIdx = sortedIndices[inCount + i];
                    m_OthersIndices.Add(originalIdx);
                }
            }

            m_OthersSorted = false;
            m_LastOthersSortCamPos = Vector3.zero;

            // Compute average outlier ring radius
            m_OutlierRingRadius = 0f;
            if (othersCount > 0 && m_AllPositionsNativeValid)
            {
                Vector3 center = m_RootBounds.center;
                double accum = 0.0;
                for (int i = 0; i < othersCount; i++)
                {
                    int orig = sortedIndices[inCount + i];
                    if (orig >= 0 && orig < m_AllPositionsNative.Length)
                    {
                        float3 p = m_AllPositionsNative[orig];
                        accum += Vector3.Distance(center, (Vector3)p);
                    }
                }
                m_OutlierRingRadius = (float)(accum / othersCount);
            }

            // Clean up sortedIndices (last remaining temp allocation)
            if (sortedIndices.IsCreated) sortedIndices.Dispose();

            // Yield before final operations
            await Task.Yield();

            // Tighten bounding boxes
            TightenBounds();

            m_Built = true;

            // Ensure GPU buffer exists
            if (m_VisibleIndicesBuffer == null)
            {
                m_VisibleIndicesBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, 1, sizeof(uint))
                {
                    name = "GaussianSplatVisibleIndices"
                };
                var init = new NativeArray<uint>(1, Allocator.Temp);
                init[0] = 0u;
                m_VisibleIndicesBuffer.SetData(init);
                visibleSplatCount = 0;
                init.Dispose();
            }

                Debug.Log($"Octree build completed: {m_Nodes.Count} total nodes, others={m_OthersIndices.Length}");

                EnsureVisibleSplatIndicesCapacity(m_TotalSplats);
            }
            finally
            {
                m_IsBuilding = false;
            }
        }

        /// <summary>
        /// Async recursive build that works directly with sorted index array slices.
        /// Uses array slicing instead of copying to reduce memory allocations.
        /// Yields periodically during deep recursion to allow GC to run.
        /// </summary>
        async Task BuildRecursiveOptimizedAsync(int nodeIndex, int depth, NativeArray<int> sortedIndices, int startIdx, int count, NativeArray<float3> positions)
        {
            var node = m_Nodes[nodeIndex];

            // Check termination conditions
            if (depth >= m_MaxDepth || count <= m_MaxSplatsPerLeaf)
            {
                // Make this a leaf node and store original indices for this leaf
                node.isLeaf = true;
                node.splatIndices.Clear();
                for (int i = 0; i < count; i++)
                {
                    int splatIdx = sortedIndices[startIdx + i];
                    // Use m_AllPositionsNative instead of positions parameter for safe access
                    if (m_AllPositionsNativeValid && (uint)splatIdx < (uint)m_AllPositionsNative.Length)
                    {
                        node.splatIndices.Add(splatIdx);
                    }
                }

                m_Nodes[nodeIndex] = node;
                return;
            }

            // Yield every 3 levels to allow GC to run (reduces peak memory)
            if (depth % 3 == 0 && depth > 0)
                await Task.Yield();

            // Create 8 child nodes
            var center = node.bounds.center;
            var size = node.bounds.size * 0.5f;

            node.childIndices.Clear();
            node.isLeaf = false;

            // Create child bounds
            var childBounds = new Bounds[8];
            for (int i = 0; i < 8; i++)
            {
                var offset = new Vector3(
                    (i & 1) != 0 ? size.x * 0.5f : -size.x * 0.5f,
                    (i & 2) != 0 ? size.y * 0.5f : -size.y * 0.5f,
                    (i & 4) != 0 ? size.z * 0.5f : -size.z * 0.5f
                );
                childBounds[i] = new Bounds(center + offset, size);
            }

            // OPTIMIZATION: Use single pre-allocated buffer for child distribution
            // This avoids creating 8 separate NativeList allocations per recursion level
            var childSplatsIdx = new NativeList<int>[8];
            for (int i = 0; i < 8; i++)
                childSplatsIdx[i] = new NativeList<int>(Allocator.Temp);

            // Validate that we can still access positions safely
            if (!m_AllPositionsNativeValid || !m_AllPositionsNative.IsCreated)
            {
                Debug.LogError("BuildRecursiveOptimizedAsync: m_AllPositionsNative is invalid during recursion");
                // Cleanup temp lists before returning
                for (int i = 0; i < 8; i++)
                {
                    childSplatsIdx[i].Dispose();
                }
                return;
            }

            // Assign splats to child nodes based on position
            // Use m_AllPositionsNative instead of positions parameter for safe access after await
            for (int ii = 0; ii < count; ii++)
            {
                int splatIdx = sortedIndices[startIdx + ii];
                if ((uint)splatIdx >= (uint)m_AllPositionsNative.Length)
                    continue;

                float3 pos = m_AllPositionsNative[splatIdx];

                int childIndex = 0;
                if (pos.x > center.x) childIndex |= 1;
                if (pos.y > center.y) childIndex |= 2;
                if (pos.z > center.z) childIndex |= 4;

                childSplatsIdx[childIndex].Add(splatIdx);
            }

            // Create a temporary buffer to copy child indices into sortedIndices array
            // This allows us to reuse sortedIndices for child recursion (in-place sorting)
            int writePos = startIdx;
            for (int i = 0; i < 8; i++)
            {
                for (int j = 0; j < childSplatsIdx[i].Length; j++)
                {
                    sortedIndices[writePos++] = childSplatsIdx[i][j];
                }
            }

            // Create child nodes and recurse using array slices
            int childStartIdx = startIdx;

            // Store child counts before disposing the temp lists
            var childCounts = new int[8];
            for (int i = 0; i < 8; i++)
            {
                childCounts[i] = childSplatsIdx[i].Length;
            }

            // Dispose all temp lists BEFORE any await calls to prevent disposed object access
            for (int i = 0; i < 8; i++)
            {
                childSplatsIdx[i].Dispose();
            }

            for (int i = 0; i < 8; i++)
            {
                int childCount = childCounts[i];

                var childNode = new OctreeNode
                {
                    bounds = childBounds[i],
                    center = childBounds[i].center,
                    splatIndices = new NativeList<int>(Allocator.Persistent),
                    childIndices = new NativeList<int>(8, Allocator.Persistent),
                    isLeaf = childCount == 0,
                    maxExtent = Mathf.Max(childBounds[i].extents.x, Mathf.Max(childBounds[i].extents.y, childBounds[i].extents.z))
                };

                int childNodeIndex = m_Nodes.Count;
                m_Nodes.Add(childNode);

                // Register child index with parent (node is a class reference, so this modifies the actual node)
                node.childIndices.Add(childNodeIndex);

                // Recursively build child only if it has splats (using array slice, no copy!)
                if (childCount > 0)
                {
                    await BuildRecursiveOptimizedAsync(childNodeIndex, depth + 1, sortedIndices, childStartIdx, childCount, m_AllPositionsNative);
                }

                childStartIdx += childCount;
            }
        }

        /// <summary>
        /// Tighten bounding boxes for all nodes based on actual splat positions.
        /// Starts from leaf nodes and propagates up to parent nodes.
        /// </summary>
        void TightenBounds()
        {
            if (m_Nodes.Count == 0)
                return;

            int tightenedNodes = 0;
            
            // Process nodes in reverse order to handle leaves first, then propagate up
            for (int i = m_Nodes.Count - 1; i >= 0; i--)
            {
                if (TightenNodeBounds(i))
                    tightenedNodes++;
            }

            Debug.Log($"Octree bounds tightened: {tightenedNodes}/{m_Nodes.Count} nodes updated");
        }

        /// <summary>
        /// Tighten the bounds of a specific node based on its splats or child bounds.
        /// </summary>
        /// <returns>True if the bounds were changed, false otherwise</returns>
        bool TightenNodeBounds(int nodeIndex)
        {
            if (nodeIndex >= m_Nodes.Count)
                return false;

            var node = m_Nodes[nodeIndex];
            var originalBounds = node.bounds;

            if (node.isLeaf)
            {
                if (node.splatIndices.IsCreated && node.splatIndices.Length > 0)
                {
                    int firstSplatIdx = node.splatIndices[0];
                    if (TryGetSplatPosition(firstSplatIdx, out float3 firstPos))
                    {
                        float3 min = firstPos;
                        float3 max = firstPos;
                        for (int i = 1; i < node.splatIndices.Length; i++)
                        {
                            int splatIdx = node.splatIndices[i];
                            if (TryGetSplatPosition(splatIdx, out float3 pos))
                            {
                                min = math.min(min, pos);
                                max = math.max(max, pos);
                            }
                        }
                        Vector3 center = (Vector3)((min + max) * 0.5f);
                        Vector3 size = (Vector3)(max - min);
                        const float minSize = 0.001f;
                        size.x = Mathf.Max(size.x, minSize);
                        size.y = Mathf.Max(size.y, minSize);
                        size.z = Mathf.Max(size.z, minSize);
                        node.bounds = new Bounds(center, size);
                        node.maxExtent = Mathf.Max(size.x, Mathf.Max(size.y, size.z)) * 0.5f;
                        m_Nodes[nodeIndex] = node;
                        return !BoundsAreEqual(originalBounds, node.bounds);
                    }
                }
                return false;
            }
            else
            {
                // For internal nodes, calculate bounds based on child node bounds
                if (node.childIndices.IsCreated && node.childIndices.Length > 0)
                {
                    bool hasValidChild = false;
                    float3 min = float3.zero;
                    float3 max = float3.zero;

                    for (int i = 0; i < node.childIndices.Length; i++)
                    {
                        int childIndex = node.childIndices[i];
                        if (childIndex < m_Nodes.Count)
                        {
                            var childNode = m_Nodes[childIndex];

                            // Only include non-empty children in bounds calculation
                            bool childHasContent = childNode.isLeaf
                                ? (childNode.splatIndices.IsCreated && childNode.splatIndices.Length > 0)
                                : (childNode.childIndices.IsCreated && childNode.childIndices.Length > 0);

                            if (childHasContent)
                            {
                                Vector3 childMin = childNode.bounds.min;
                                Vector3 childMax = childNode.bounds.max;

                                if (!hasValidChild)
                                {
                                    min = (float3)childMin;
                                    max = (float3)childMax;
                                    hasValidChild = true;
                                }
                                else
                                {
                                    min = math.min(min, (float3)childMin);
                                    max = math.max(max, (float3)childMax);
                                }
                            }
                        }
                    }

                    // Update bounds if we found valid children
                    if (hasValidChild)
                    {
                        Vector3 center = (Vector3)((min + max) * 0.5f);
                        Vector3 size = (Vector3)(max - min);
                        
                        // Ensure minimum size to avoid zero-size bounds
                        const float minSize = 0.001f;
                        size.x = Mathf.Max(size.x, minSize);
                        size.y = Mathf.Max(size.y, minSize);
                        size.z = Mathf.Max(size.z, minSize);

                        node.bounds = new Bounds(center, size);
                        node.maxExtent = Mathf.Max(size.x, Mathf.Max(size.y, size.z)) * 0.5f;

                        // Update the node in the list
                        m_Nodes[nodeIndex] = node;
                        
                        // Check if bounds actually changed
                        return !BoundsAreEqual(originalBounds, node.bounds);
                    }
                }
                return false; // No valid children, bounds unchanged
            }
        }

        /// <summary>
        /// Helper method to compare two bounds for equality with small tolerance.
        /// </summary>
        bool BoundsAreEqual(Bounds a, Bounds b)
        {
            const float tolerance = 1e-6f;
            return Vector3.Distance(a.center, b.center) < tolerance && 
                   Vector3.Distance(a.size, b.size) < tolerance;
        }

        void UpdateVisibleIndicesBuffer()
        {
            if (visibleSplatCount == 0)
                return;

            if (!m_VisibleSplatIndicesValid || !m_VisibleSplatIndices.IsCreated)
            {
                Debug.LogWarning("Visible splat indices native array is invalid during buffer update");
                return;
            }

            // Ensure buffer is large enough
            int requiredSize = visibleSplatCount;
            if (m_VisibleIndicesBuffer == null || m_VisibleIndicesBuffer.count < requiredSize)
            {
                m_VisibleIndicesBuffer?.Dispose();
                // Allocate with some extra space to avoid frequent reallocations
                int bufferSize = Mathf.NextPowerOfTwo(requiredSize);
                m_VisibleIndicesBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, bufferSize, sizeof(uint))
                {
                    name = "GaussianSplatVisibleIndices"
                };
            }

            // Upload visible indices directly from native array (reinterpret cast from int to uint)
            unsafe
            {
                // Create a NativeArray<uint> view of our int data (reinterpret cast)
                var uintView = NativeArrayUnsafeUtility.ConvertExistingDataToNativeArray<uint>(
                    (void*)m_VisibleSplatIndices.GetUnsafeReadOnlyPtr(),
                    visibleSplatCount,
                    Allocator.None);
                
                #if ENABLE_UNITY_COLLECTIONS_CHECKS
                NativeArrayUnsafeUtility.SetAtomicSafetyHandle(ref uintView, 
                    NativeArrayUnsafeUtility.GetAtomicSafetyHandle(m_VisibleSplatIndices));
                #endif
                
                m_VisibleIndicesBuffer.SetData(uintView, 0, 0, visibleSplatCount);
            }
        }

        /// <summary>
        /// Get debug information about octree structure.
        /// </summary>
        public void GetDebugInfo(out int leafNodes, out int maxDepthReached, out int maxSplatsInLeaf)
        {
            leafNodes = 0;
            maxDepthReached = 0;
            maxSplatsInLeaf = 0;

            GetDebugInfoRecursive(0, 0, ref leafNodes, ref maxDepthReached, ref maxSplatsInLeaf);
        }

        void GetDebugInfoRecursive(int nodeIndex, int depth, ref int leafNodes, ref int maxDepth, ref int maxSplats)
        {
            if (nodeIndex >= m_Nodes.Count)
                return;

            var node = m_Nodes[nodeIndex];
            maxDepth = Mathf.Max(maxDepth, depth);

            if (node.isLeaf)
            {
                // Only count non-empty leaves
                if (node.splatIndices.IsCreated && node.splatIndices.Length > 0)
                {
                    leafNodes++;
                    maxSplats = Mathf.Max(maxSplats, node.splatIndices.Length);
                }
            }
            else
            {
                // Traverse registered child indices
                if (node.childIndices.IsCreated)
                {
                    for (int i = 0; i < node.childIndices.Length; i++)
                    {
                        int childIndex = node.childIndices[i];
                        if (childIndex < m_Nodes.Count)
                        {
                            GetDebugInfoRecursive(childIndex, depth + 1, ref leafNodes, ref maxDepth, ref maxSplats);
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Draw wireframe boxes for each non-empty leaf node. Call this from a MonoBehaviour's OnDrawGizmos or OnDrawGizmosSelected.
        /// </summary>
        public void DrawLeafBoundsGizmos(Color color)
        {
            if (!m_Built || m_Nodes.Count == 0)
                return;

            var prev = Gizmos.color;
            Gizmos.color = color;

            for (int i = 0; i < m_Nodes.Count; i++)
            {
                var node = m_Nodes[i];
                if (!node.isLeaf)
                    continue;

                // Skip empty leaves
                if (!node.splatIndices.IsCreated || node.splatIndices.Length <= 0)
                    continue;

                Gizmos.DrawWireCube(node.bounds.center, node.bounds.size);
            }

            Gizmos.color = prev;
        }

        public void Clear()
        {
            // Complete any pending sort job before cleanup
            if (m_SortJobRunning)
            {
                m_SortJobHandle.Complete();
                m_SortJobRunning = false;
            }

            // Dispose job data (but not slices of cached arrays)
            if (m_NodesToSort.IsCreated)
            {
                try { m_NodesToSort.Dispose(); } catch {}
            }

            // Dispose all node native buffers
            for (int i = 0; i < m_Nodes.Count; i++)
            {
                try { m_Nodes[i].Dispose(); } catch {}
            }
            m_Nodes.Clear();

            // Dispose outlier indices
            if (m_OthersIndicesValid && m_OthersIndices.IsCreated)
            {
                try { m_OthersIndices.Dispose(); } catch {}
                m_OthersIndicesValid = false;
            }
            if (m_VisibleSplatIndicesValid && m_VisibleSplatIndices.IsCreated)
            {
                try { m_VisibleSplatIndices.Dispose(); } catch {}
                m_VisibleSplatIndicesValid = false;
            }
            if (m_VisibleNodeRefsValid && m_VisibleNodeRefs.IsCreated)
            {
                try { m_VisibleNodeRefs.Dispose(); } catch {}
                m_VisibleNodeRefsValid = false;
            }
            m_TraversalStack.Clear(); // Clear the reusable stack
            m_VisibleIndicesBuffer?.Dispose();
            m_VisibleIndicesBuffer = null;
            m_DistanceSortArray = null; // Release sort array memory
            visibleSplatCount = 0;
            m_Built = false;
            m_OthersSorted = false;
            m_LastOthersSortCamPos = Vector3.zero;
            m_OutlierRingRadius = 0f;

            if (m_AllPositionsNativeValid && m_AllPositionsNative.IsCreated)
            {
                m_AllPositionsNative.Dispose();
                m_AllPositionsNativeValid = false;
            }

            // Dispose pooled job arrays (optimization 4)
            if (m_JobArraysValid)
            {
                if (m_CachedNodeSplatListPointers.IsCreated)
                {
                    try { m_CachedNodeSplatListPointers.Dispose(); } catch {}
                }
                if (m_CachedNodeRanges.IsCreated)
                {
                    try { m_CachedNodeRanges.Dispose(); } catch {}
                }
                m_JobArraysValid = false;
            }

            m_TotalSplats = 0;
        }

        public void Dispose()
        {
            Clear();
        }

        /// <summary>
        /// Sort visible splat indices by 3D distance from camera (front-to-back for alpha blending).
        /// Hierarchical sorting optimization.
        /// Fully async - uses previous frame's sorted data while current frame's sort runs in background.
        /// </summary>
        public void SortVisibleSplatsByDepth(Camera camera)
        {
            if (!m_Built)
                return;
            var camPosition = camera.transform.position;
            var camRotation = camera.transform.rotation;

            if (!m_VisibleSplatIndicesValid || !m_VisibleSplatIndices.IsCreated)
            {
                visibleSplatCount = 0;
                return;
            }

            // Early exit: skip frustum culling if camera hasn't moved much
            // This is a huge optimization - frustum culling is expensive!
            bool needsCulling = true;
            float positionDelta = Vector3.Distance(camPosition, m_LastCullCameraPosition);
            float rotationDelta = Quaternion.Angle(camRotation, m_LastCullCameraRotation);

            if (positionDelta < m_CullUpdateThreshold && rotationDelta < 0.5f)
            {
                // Camera barely moved, skip expensive culling but still check for completed sort jobs
                needsCulling = false;
            }

            if (needsCulling)
            {
                // Ensure m_VisibleNodeRefs is created
                if (!m_VisibleNodeRefsValid || !m_VisibleNodeRefs.IsCreated)
                {
                    if (m_VisibleNodeRefsValid && m_VisibleNodeRefs.IsCreated)
                        m_VisibleNodeRefs.Dispose();
                    m_VisibleNodeRefs = new NativeList<VisibleNodeRef>(256, Allocator.Persistent);
                    m_VisibleNodeRefsValid = true;
                }
                else
                {
                    m_VisibleNodeRefs.Clear();
                }

                // Calculate frustum planes into reusable array to avoid GC allocation
                GeometryUtility.CalculateFrustumPlanes(camera, m_FrustumPlanes);
                CollectVisibleNodesWithDistance(0, m_FrustumPlanes, camPosition);

                m_LastCullCameraPosition = camPosition;
                m_LastCullCameraRotation = camRotation;
            }

            // Sort node references by distance (front-to-back) using Burst-optimized sort
            // Only sort if we did culling (otherwise list is unchanged)
            if (needsCulling && m_VisibleNodeRefs.Length > 1)
            {
                // Use native sorting for better performance
                m_VisibleNodeRefs.AsArray().Sort(new VisibleNodeRefDistanceComparer());
            }

            // ASYNC STRATEGY (fully non-blocking):
            // Frame N:   Build indices from node data (sorted in frame N-1 or earlier)
            //            Check if job from frame N-1 completed (non-blocking)
            //            If completed: apply results to nodes (ready for frame N+1)
            //            If no job running: schedule new sort job for current camera position
            // Frame N+1: Will use the sorted data from frame N's job
            //
            // This creates 1-2 frame latency for sorting but NEVER blocks the main thread.
            // The slight latency is acceptable since sorting is a rendering optimization.

            // Build visible splat indices from CURRENT node data
            // Skip if culling was skipped (nothing changed)
            if (needsCulling)
            {
                int currentIndex = 0;

                // First, add node splats (front elements for front-to-back rendering)
                for (int i = 0; i < m_VisibleNodeRefs.Length; i++)
                {
                    var nodeRef = m_VisibleNodeRefs[i];
                    var node = m_Nodes[nodeRef.nodeIndex];
                    if (node != null && node.splatIndices.IsCreated && node.splatIndices.Length > 0)
                    {
                        // Ensure we have enough space
                        if (currentIndex + node.splatIndices.Length > m_VisibleSplatIndices.Length)
                        {
                            if (!m_VisibleSplatIndicesValid || !m_VisibleSplatIndices.IsCreated)
                            {
                                visibleSplatCount = currentIndex;
                                if (m_BufferNeedsUpload || m_LastUploadedSplatCount != visibleSplatCount)
                                {
                                    UpdateVisibleIndicesBuffer();
                                    m_LastUploadedSplatCount = visibleSplatCount;
                                    m_BufferNeedsUpload = false;
                                }
                                return;
                            }
                        }

                        // Copy node splat indices using bulk memory copy (much faster than loop)
                        NativeArray<int>.Copy(node.splatIndices.AsArray(), 0, m_VisibleSplatIndices, currentIndex, node.splatIndices.Length);
                        currentIndex += node.splatIndices.Length;
                    }
                }

                // Finally, add outliers (background elements for front-to-back rendering)
                if (m_OthersIndices.Length > 0)
                {
                    if (currentIndex + m_OthersIndices.Length > m_VisibleSplatIndices.Length)
                    {
                        EnsureVisibleSplatIndicesCapacity(currentIndex + m_OthersIndices.Length);
                        if (!m_VisibleSplatIndicesValid || !m_VisibleSplatIndices.IsCreated)
                        {
                            visibleSplatCount = currentIndex;
                            if (m_BufferNeedsUpload || m_LastUploadedSplatCount != visibleSplatCount)
                            {
                                UpdateVisibleIndicesBuffer();
                                m_LastUploadedSplatCount = visibleSplatCount;
                                m_BufferNeedsUpload = false;
                            }
                            return;
                        }
                    }

                    // Copy outliers using bulk memory copy (much faster than loop)
                    NativeArray<int>.Copy(m_OthersIndices.AsArray(), 0, m_VisibleSplatIndices, currentIndex, m_OthersIndices.Length);
                    currentIndex += m_OthersIndices.Length;
                }

                visibleSplatCount = currentIndex;
                m_BufferNeedsUpload = true; // Mark for upload since we rebuilt the list
            }

            // Complete previous job if it's ready (non-blocking check)
            // This applies the sorted results to the node data
            bool sortCompleted = false;
            if (m_SortJobRunning && m_SortJobHandle.IsCompleted)
            {
                CompleteSortJob(m_SortJobCameraPosition);
                sortCompleted = true;
                m_BufferNeedsUpload = true; // Data changed, need GPU upload
            }

            // Only upload to GPU if data changed or count changed
            if (m_BufferNeedsUpload || m_LastUploadedSplatCount != visibleSplatCount)
            {
                UpdateVisibleIndicesBuffer();
                m_LastUploadedSplatCount = visibleSplatCount;
                m_BufferNeedsUpload = false;
            }

            // Only schedule a new job if no job is currently running
            // This prevents overlapping jobs and resource conflicts
            if (!m_SortJobRunning)
            {
                ScheduleBurstSortJobs(camPosition);
            }
        }

        bool ShouldResortOutliers(Vector3 camPosition)
        {
            if (m_OthersIndices.Length == 0)
                return false;
            if (!m_OthersSorted)
                return true;
            float baseThreshold = Mathf.Max(minOutlierResortDistance, m_OutlierRingRadius * outlierResortMoveFraction);
            float sqMove = (camPosition - m_LastOthersSortCamPos).sqrMagnitude;
            return sqMove >= baseThreshold * baseThreshold;
        }

        public void SetOutlierResortFraction(float fraction, float minDistance = 0.05f)
        {
            outlierResortMoveFraction = Mathf.Max(0f, fraction);
            minOutlierResortDistance = Mathf.Max(0f, minDistance);
        }

        /// <summary>
        /// Schedule Burst-compiled parallel sorting jobs for all visible nodes and outliers.
        /// Uses IJobParallelFor for true parallel execution across CPU cores.
        /// Sorts data in-place within NativeLists using optimized hybrid sort - zero GC allocation.
        /// Optimized with pooled arrays and work-weighted batch sizing (optimizations 4 & 5).
        /// </summary>
        void ScheduleBurstSortJobs(Vector3 camPosition)
        {
            if (!m_AllPositionsNativeValid)
            {
                Debug.LogWarning("Cannot schedule Burst sort jobs: global positions buffer is invalid");
                return;
            }

            // Collect nodes that need sorting
            m_NodesToSort = new NativeList<int>(m_VisibleNodeRefs.Length, Allocator.Persistent);

            for (int i = 0; i < m_VisibleNodeRefs.Length; i++)
            {
                var nodeRef = m_VisibleNodeRefs[i];
                var node = m_Nodes[nodeRef.nodeIndex];

                if (!node.splatIndices.IsCreated || node.splatIndices.Length <= 1)
                    continue;

                // Check if already sorted for this camera direction
                bool needsSort = !node.isSorted;
                if (!needsSort)
                {
                    Vector3 nodeCenter = node.bounds.center;
                    Vector3 oldDirection = (nodeCenter - node.lastSortCameraPosition).normalized;
                    Vector3 newDirection = (nodeCenter - oldDirection * node.maxExtent - camPosition).normalized;
                    float cosineAngle = Vector3.Dot(oldDirection, newDirection);
                    needsSort = cosineAngle < sortDirectionThreshold;
                }

                if (needsSort)
                {
                    m_NodesToSort.Add(nodeRef.nodeIndex);
                }
            }

            // Include outliers if needed
            m_SortOutliers = ShouldResortOutliers(camPosition) && m_OthersIndices.Length > 1;
            int totalJobCount = m_NodesToSort.Length + (m_SortOutliers ? 1 : 0);

            if (totalJobCount == 0)
            {
                m_NodesToSort.Dispose();
                return;
            }

            // Use pooled arrays instead of allocating new ones (optimization 4)
            EnsureJobArraysCapacity(totalJobCount);

            if (!m_JobArraysValid)
            {
                Debug.LogError("Failed to allocate job arrays for sorting");
                m_NodesToSort.Dispose();
                return;
            }

            // Reuse cached arrays by creating slices
            m_NodeSplatListPointers = m_CachedNodeSplatListPointers.GetSubArray(0, totalJobCount);
            m_NodeRanges = m_CachedNodeRanges.GetSubArray(0, totalJobCount);

            // Setup node pointers and ranges for in-place sorting
            // Also calculate total work for batch size optimization (optimization 5)
            int totalWork = 0;

            for (int i = 0; i < m_NodesToSort.Length; i++)
            {
                int nodeIndex = m_NodesToSort[i];
                var node = m_Nodes[nodeIndex];
                int nodeCount = node.splatIndices.Length;

                // Get unsafe pointer to the node's NativeList directly from the list
                unsafe
                {
                    m_NodeSplatListPointers[i] = *node.splatIndices.GetUnsafeList();
                }

                // Store range metadata
                m_NodeRanges[i] = new GaussianSplatBurstSorting.NodeSortRange
                {
                    nodeIndex = i,
                    length = nodeCount
                };

                totalWork += nodeCount;
            }

            // Add outliers if needed
            if (m_SortOutliers)
            {
                unsafe
                {
                    m_NodeSplatListPointers[m_NodesToSort.Length] = *m_OthersIndices.GetUnsafeList();
                }

                m_NodeRanges[m_NodesToSort.Length] = new GaussianSplatBurstSorting.NodeSortRange
                {
                    nodeIndex = m_NodesToSort.Length,
                    length = m_OthersIndices.Length
                };

                totalWork += m_OthersIndices.Length;
            }

            // Schedule the parallel hybrid sort job
            var parallelSortJob = new GaussianSplatBurstSorting.RadixSortMultipleNodesJob
            {
                nodeSplatLists = m_NodeSplatListPointers,
                nodeRanges = m_NodeRanges,
                allPositions = m_AllPositionsNative,
                cameraPosition = (float3)camPosition
            };

            // Calculate optimal batch size using work-weighted approach (optimization 5)
            // Target: Balance between job overhead and load balancing
            // Aim for ~2000-5000 splats per batch for good CPU utilization
            int workerCount = Unity.Jobs.LowLevel.Unsafe.JobsUtility.JobWorkerCount;
            int targetSplatsPerBatch = 3000;
            int idealBatchCount = Mathf.Max(workerCount, totalWork / targetSplatsPerBatch);
            int batchSize = Mathf.Max(1, totalJobCount / idealBatchCount);

            // Clamp batch size to reasonable bounds
            batchSize = Mathf.Clamp(batchSize, 1, 32);

            // Schedule with parallel execution - DON'T complete yet
            m_SortJobHandle = parallelSortJob.Schedule(totalJobCount, batchSize);
            m_SortJobRunning = true;
            m_SortJobCameraPosition = camPosition; // Store for next frame's completion
        }

        /// <summary>
        /// Complete the pending sort job and mark nodes as sorted.
        /// Data is already sorted in-place, so we just need to update metadata.
        /// </summary>
        void CompleteSortJob(Vector3 camPosition)
        {
            if (!m_SortJobRunning)
                return;

            // Wait for job to complete
            m_SortJobHandle.Complete();
            m_SortJobRunning = false;

            // Mark nodes as sorted (data is already sorted in-place)
            for (int i = 0; i < m_NodesToSort.Length; i++)
            {
                int nodeIndex = m_NodesToSort[i];
                var node = m_Nodes[nodeIndex];

                // Mark as sorted
                node.isSorted = true;
                node.lastSortCameraPosition = camPosition;
            }

            if (m_SortOutliers)
            {
                m_OthersSorted = true;
                m_LastOthersSortCamPos = camPosition;
            }

            // Cleanup job data (but keep pooled arrays for reuse - optimization 4)
            // m_NodeSplatListPointers and m_NodeRanges are slices of cached arrays, don't dispose
            if (m_NodesToSort.IsCreated)
                m_NodesToSort.Dispose();
        }

        /// <summary>
        /// Sort splats in a node and mark it as sorted for the current camera view.
        /// </summary>
        public bool SortNodeSplats(int nodeIndex, Vector3 camPosition, bool forceSort = false)
        {
            if (nodeIndex < 0 || nodeIndex >= m_Nodes.Count) return false;
            var node = m_Nodes[nodeIndex];
            if (!node.splatIndices.IsCreated || node.splatIndices.Length <= 1) return false;

            if (!forceSort && node.isSorted)
            {
                Vector3 nodeCenter = node.bounds.center;
                Vector3 oldDirection = (nodeCenter - node.lastSortCameraPosition).normalized;
                Vector3 newDirection = (nodeCenter - oldDirection * node.maxExtent - camPosition).normalized;

                float cosineAngle = Vector3.Dot(oldDirection, newDirection);
                if (cosineAngle >= sortDirectionThreshold)
                    return false;
            }

            SortSplatsInNode(node.splatIndices.AsArray(), camPosition);
            node.isSorted = true;
            node.lastSortCameraPosition = camPosition;
            return true;
        }

        /// <summary>
        /// Mark all nodes and outliers as needing re-sort.
        /// </summary>
        public void InvalidateAllSorts()
        {
            for (int i = 0; i < m_Nodes.Count; i++)
            {
                m_Nodes[i].isSorted = false;
            }
            m_OthersSorted = false;
        }

        /// <summary>
        /// Set the sort direction threshold using angle in degrees for easier configuration.
        /// </summary>
        public void SetSortDirectionThresholdDegrees(float angleDegrees)
        {
            sortDirectionThreshold = Mathf.Cos(angleDegrees * Mathf.Deg2Rad);
        }

        void SortSplatsInNode(NativeArray<int> splatIndices, Vector3 camPosition)
        {
            int count = splatIndices.Length;
            if (count <= 1) return;
            if (m_DistanceSortArray == null || m_DistanceSortArray.Length < count)
                m_DistanceSortArray = new (float distance, int index)[Mathf.NextPowerOfTwo(count)];
            for (int i = 0; i < count; i++)
            {
                int originalSplatIdx = splatIndices[i];
                if (TryGetSplatPosition(originalSplatIdx, out float3 splatPos))
                {
                    float distance = ((Vector3)splatPos - camPosition).sqrMagnitude;
                    m_DistanceSortArray[i] = (distance, originalSplatIdx);
                }
                else
                {
                    m_DistanceSortArray[i] = (0f, originalSplatIdx);
                }
            }
            System.Array.Sort(m_DistanceSortArray, 0, count, System.Collections.Generic.Comparer<(float distance, int index)>.Create((a, b) => a.distance.CompareTo(b.distance))); // Front-to-back
            for (int i = 0; i < count; i++)
                splatIndices[i] = m_DistanceSortArray[i].index;
        }

        void CollectVisibleNodesWithDistance(int nodeIndex, Plane[] frustumPlanes, Vector3 camPosition)
        {
            m_TraversalStack.Clear();
            
            // Early exit if invalid starting node
            if (nodeIndex >= m_Nodes.Count)
                return;
                
            m_TraversalStack.Push(nodeIndex);
            
            while (m_TraversalStack.Count > 0)
            {
                int currentNodeIndex = m_TraversalStack.Pop();
                
                // Bounds check
                if (currentNodeIndex >= m_Nodes.Count)
                    continue;
                    
                var node = m_Nodes[currentNodeIndex];
                
                // Frustum culling - early exit if node not visible
                if (!GeometryUtility.TestPlanesAABB(frustumPlanes, node.bounds))
                    continue;
                
                if (node.isLeaf)
                {
                    // Add leaf node if it has splats
                    if (node.splatIndices.IsCreated && node.splatIndices.Length > 0)
                    {
                        float nodeDistance = (node.center - camPosition).sqrMagnitude;
                        m_VisibleNodeRefs.Add(new VisibleNodeRef
                        {
                            distance = nodeDistance,
                            nodeIndex = currentNodeIndex
                        });
                    }
                }
                else if (node.childIndices.IsCreated)
                {
                    // Add children to stack for traversal (reverse order for consistent traversal)
                    for (int i = node.childIndices.Length - 1; i >= 0; i--)
                    {
                        int childIndex = node.childIndices[i];
                        if (childIndex >= 0 && childIndex < m_Nodes.Count)
                        {
                            var childNode = m_Nodes[childIndex];
                            if (childNode != null)
                            {
                                // Only traverse children that have content or are internal nodes
                                if ((childNode.splatIndices.IsCreated && childNode.splatIndices.Length > 0) || !childNode.isLeaf)
                                {
                                    m_TraversalStack.Push(childIndex);
                                }
                            }
                        }
                    }
                }
            }
        }

    }
}
