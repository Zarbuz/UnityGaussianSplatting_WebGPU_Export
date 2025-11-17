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
        NativeArray<int> m_FlattenedSplatIndices;
        NativeArray<GaussianSplatBurstSorting.NodeSortRange> m_NodeRanges;
        NativeList<int> m_NodesToSort;
        bool m_SortOutliers;
		Vector3 m_SortJobCameraPosition;

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
        /// Build octree from splat position data and bounds.
        /// </summary>
        public void Build(NativeArray<float3> splatPositions, Bounds sceneBounds, float splatPercent)
        {
            Clear();
            // m_OthersNodeIndex removed - use m_OthersIndices list instead

            if (splatPositions.Length == 0)
            {
                Debug.LogWarning("GaussianSplatOctree.Build: No splat positions provided");
                return;
            }

            Debug.Log($"Building octree with {splatPositions.Length} splats, bounds: {sceneBounds}");

            // Compute center of mass and identify 95% closest splats
            int total = splatPositions.Length;
            m_TotalSplats = total;
            float3 com = float3.zero;
            for (int i = 0; i < total; i++)
                com += splatPositions[i];
            com /= total;

            var distList = new List<(int idx, float d)>(total);
            for (int i = 0; i < total; i++)
            {
                float distance = math.distance(splatPositions[i], com);
                distList.Add((i, distance));
            }
            distList.Sort((a, b) => a.d.CompareTo(b.d));

            // Reorder m_SplatInfos so that the closest part are first, others last
            int inCount = Mathf.CeilToInt(total * splatPercent);
            inCount = Mathf.Clamp(inCount, 1, total);
            int othersCount = total - inCount;

            // Local build-time splat info list
            var splatInfos = new List<SplatInfo>(total);
            for (int i = 0; i < total; i++)
            {
                int src = distList[i].idx;
                splatInfos.Add(new SplatInfo { position = splatPositions[src], originalIndex = src });
            }

            // Create / update global native positions buffer
            if (m_AllPositionsNativeValid)
            {
                if (m_AllPositionsNative.IsCreated) m_AllPositionsNative.Dispose();
                m_AllPositionsNativeValid = false;
            }
            try
            {
                m_AllPositionsNative = new NativeArray<float3>(total, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
                for (int i = 0; i < splatInfos.Count; i++)
                {
                    var si = splatInfos[i];
                    int orig = si.originalIndex;
                    if ((uint)orig < (uint)total)
                        m_AllPositionsNative[orig] = si.position;
                }
                m_AllPositionsNativeValid = true;
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"Failed to allocate global native positions buffer: {ex.Message}");
                if (m_AllPositionsNative.IsCreated) m_AllPositionsNative.Dispose();
                m_AllPositionsNativeValid = false;
            }

            // Create root bounds based on the inCount splats (centered on center-of-mass)
            Bounds rootBounds;
            if (inCount > 0)
            {
                float3 min = splatInfos[0].position;
                float3 max = splatInfos[0].position;
                for (int i = 1; i < inCount; i++)
                {
                    min = math.min(min, splatInfos[i].position);
                    max = math.max(max, splatInfos[i].position);
                }
                rootBounds = new Bounds((max + min) * 0.5f, max - min);
            }
            else
            {
                // Fallback to provided scene bounds
                rootBounds = sceneBounds;
            }

            m_RootBounds = rootBounds;

            // Build tree recursively using only the in-root splats
            m_Nodes.Clear();

            // Create root node covering the in-root splats
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

            // Build recursively starting from root (only for the in-root partition)
            var rootSplatList = new NativeList<int>(inCount, Allocator.Temp);
            for (int i = 0; i < inCount; i++) rootSplatList.Add(i); // indices into splatInfos
            BuildRecursive(0, 0, rootSplatList, splatInfos);
            rootSplatList.Dispose();

            // Handle remaining outliers: put their original indices into m_SplatIndices and track them in m_OthersIndices
            // Ensure m_OthersIndices is created before clearing
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
                    int orig = splatInfos[inCount + i].originalIndex;
                    m_OthersIndices.Add(orig);
                }
            }
            m_OthersSorted = false; // reset outlier sorting state after build
            m_LastOthersSortCamPos = Vector3.zero;
            // Compute average outlier ring radius (ignore min/max & extra stats for simplicity)
            m_OutlierRingRadius = 0f;
            if (othersCount > 0 && m_AllPositionsNativeValid)
            {
                Vector3 center = m_RootBounds.center;
                double accum = 0.0;
                for (int i = 0; i < othersCount; i++)
                {
                    int orig = splatInfos[inCount + i].originalIndex;
                    if (orig >= 0 && orig < m_AllPositionsNative.Length)
                    {
                        float3 p = m_AllPositionsNative[orig];
                        accum += Vector3.Distance(center, (Vector3)p);
                    }
                }
                m_OutlierRingRadius = (float)(accum / othersCount);
            }

            // Tighten bounding boxes starting from leaves and propagating up
            TightenBounds();

            m_Built = true;

            // Ensure a GPU buffer exists even if there are no visible splats yet.
            // Allocate a minimal 1-entry structured buffer so renderer code can safely bind/check it.
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

        void BuildRecursive(int nodeIndex, int depth, NativeList<int> splatList, List<SplatInfo> splatInfos)
        {
            var node = m_Nodes[nodeIndex];

            // Check termination conditions
            if (depth >= m_MaxDepth || splatList.Length <= m_MaxSplatsPerLeaf)
            {
                // Make this a leaf node and store original indices for this leaf
                node.isLeaf = true;
                node.splatIndices.Clear();
                for (int i = 0; i < splatList.Length; i++)
                {
                    int infoIdx = splatList[i];
                    if (infoIdx < 0 || infoIdx >= splatInfos.Count)
                    {
                        Debug.LogError($"Octree leaf node splat info index out of bounds: {infoIdx} >= {splatInfos.Count}");
                        continue;
                    }
                    node.splatIndices.Add(splatInfos[infoIdx].originalIndex);
                }

                m_Nodes[nodeIndex] = node;
                return;
            }

            // Create 8 child nodes
            var center = node.bounds.center;
            var size = node.bounds.size * 0.5f;

            node.childIndices.Clear();
            node.isLeaf = false;
            m_Nodes[nodeIndex] = node;

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

            // Distribute splats to children using NativeList
            var childSplatsIdx = new NativeList<int>[8];
            for (int i = 0; i < 8; i++)
                childSplatsIdx[i] = new NativeList<int>(Allocator.Temp);

            // Assign splats (using splatList which holds indices into splatInfos) to child nodes
            for (int ii = 0; ii < splatList.Length; ii++)
            {
                int infoIdx = splatList[ii];
                if (infoIdx < 0 || infoIdx >= splatInfos.Count)
                {
                    Debug.LogError($"Octree splat distribution info index out of bounds: {infoIdx} >= {splatInfos.Count}");
                    continue;
                }

                var splat = splatInfos[infoIdx];

                int childIndex = 0;
                if (splat.position.x > center.x) childIndex |= 1;
                if (splat.position.y > center.y) childIndex |= 2;
                if (splat.position.z > center.z) childIndex |= 4;

                childSplatsIdx[childIndex].Add(infoIdx);
            }

            // Create child nodes
            for (int i = 0; i < 8; i++)
            {
                var childNode = new OctreeNode
                {
                    bounds = childBounds[i],
                    center = childBounds[i].center,
                    splatIndices = new NativeList<int>(Allocator.Persistent),
                    childIndices = new NativeList<int>(8, Allocator.Persistent),
                    isLeaf = childSplatsIdx[i].Length == 0,
                    maxExtent = Mathf.Max(childBounds[i].extents.x, Mathf.Max(childBounds[i].extents.y, childBounds[i].extents.z))
                };

                int childNodeIndex = m_Nodes.Count;
                m_Nodes.Add(childNode);

                // Register child index with parent
                node.childIndices.Add(childNodeIndex);
                // Update parent reference in the global list (node is a reference type)
                m_Nodes[nodeIndex] = node;

                // Recursively build child only if it has splats
                if (childSplatsIdx[i].Length > 0)
                {
                    BuildRecursive(childNodeIndex, depth + 1, childSplatsIdx[i], splatInfos);
                }

                // Dispose temp child list
                childSplatsIdx[i].Dispose();
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

            // Dispose job data
            if (m_FlattenedSplatIndices.IsCreated)
            {
                try { m_FlattenedSplatIndices.Dispose(); } catch {}
            }
            if (m_NodeRanges.IsCreated)
            {
                try { m_NodeRanges.Dispose(); } catch {}
            }
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

            m_TotalSplats = 0;
        }

        public void Dispose()
        {
            Clear();
        }

        /// <summary>
        /// Sort visible splat indices by 3D distance from camera (front-to-back for alpha blending).
        /// Hierarchical sorting optimization.
        /// </summary>
        public void SortVisibleSplatsByDepth(Camera camera)
        {
            if (!m_Built)
                return;
            var camPosition = camera.transform.position;
            
            if (!m_VisibleSplatIndicesValid || !m_VisibleSplatIndices.IsCreated)
            {
                visibleSplatCount = 0;
                return;
            }

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
            var frustumPlanes = GeometryUtility.CalculateFrustumPlanes(camera);
            CollectVisibleNodesWithDistance(0, frustumPlanes, camPosition);

            // Sort node references by distance (front-to-back)
            m_VisibleNodeRefs.AsArray().Sort(new VisibleNodeRefDistanceComparer());

            // Complete the PREVIOUS frame's sort job if one is running
            // This gives the job maximum time to complete in the background
            if (m_SortJobRunning)
            {
                CompleteSortJob(m_SortJobCameraPosition);
            }

            // Schedule a NEW sort job for this frame (will be completed next frame)
            // This creates a 1-frame latency but eliminates blocking
            ScheduleBurstSortJobs(camPosition);

            // NOTE: We DON'T complete the job here - it will run in the background
            // and be completed on the next frame, giving it maximum time to execute

            // Append nodes in distance order (their lists now internally sorted and persistent)
            int currentIndex = 0;

            // First, add node splats (front elements for front-to-back rendering)
            for (int i = 0; i < m_VisibleNodeRefs.Length; i++)
            {
                var nodeRef = m_VisibleNodeRefs[i];
                var node = m_Nodes[nodeRef.nodeIndex];
                if (node.splatIndices.IsCreated && node.splatIndices.Length > 0)
                {
                    // Ensure we have enough space
                    if (currentIndex + node.splatIndices.Length > m_VisibleSplatIndices.Length)
                    {
                        if (!m_VisibleSplatIndicesValid || !m_VisibleSplatIndices.IsCreated)
                        {
                            visibleSplatCount = currentIndex;
                            UpdateVisibleIndicesBuffer();
                            return;
                        }
                    }

                    // Copy node splat indices
                    for (int j = 0; j < node.splatIndices.Length; j++)
                    {
                        m_VisibleSplatIndices[currentIndex + j] = node.splatIndices[j];
                    }
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
                        UpdateVisibleIndicesBuffer();
                        return;
                    }
                }

                for (int i = 0; i < m_OthersIndices.Length; i++)
                {
                    m_VisibleSplatIndices[currentIndex + i] = m_OthersIndices[i];
                }
                currentIndex += m_OthersIndices.Length;
            }
            
            visibleSplatCount = currentIndex;
            UpdateVisibleIndicesBuffer();
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
        /// Sorts data in-place within NativeLists - zero GC allocation.
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

            // Calculate total size needed for flattened array
            int totalSplatCount = 0;
            for (int i = 0; i < m_NodesToSort.Length; i++)
            {
                int nodeIndex = m_NodesToSort[i];
                var node = m_Nodes[nodeIndex];
                totalSplatCount += node.splatIndices.Length;
            }
            if (m_SortOutliers)
            {
                totalSplatCount += m_OthersIndices.Length;
            }

            // Create flattened array and range metadata (persist until job completes)
            m_FlattenedSplatIndices = new NativeArray<int>(totalSplatCount, Allocator.Persistent);
            m_NodeRanges = new NativeArray<GaussianSplatBurstSorting.NodeSortRange>(totalJobCount, Allocator.Persistent);

            // Flatten all node arrays into a single contiguous array
            int currentOffset = 0;
            for (int i = 0; i < m_NodesToSort.Length; i++)
            {
                int nodeIndex = m_NodesToSort[i];
                var node = m_Nodes[nodeIndex];
                int length = node.splatIndices.Length;

                // Copy node's splat indices to flattened array
                NativeArray<int>.Copy(node.splatIndices.AsArray(), 0, m_FlattenedSplatIndices, currentOffset, length);

                // Store range metadata
                m_NodeRanges[i] = new GaussianSplatBurstSorting.NodeSortRange
                {
                    offset = currentOffset,
                    length = length
                };

                currentOffset += length;
            }

            // Add outliers if needed
            if (m_SortOutliers)
            {
                int length = m_OthersIndices.Length;
                NativeArray<int>.Copy(m_OthersIndices.AsArray(), 0, m_FlattenedSplatIndices, currentOffset, length);

                m_NodeRanges[m_NodesToSort.Length] = new GaussianSplatBurstSorting.NodeSortRange
                {
                    offset = currentOffset,
                    length = length
                };
            }

            // Schedule the parallel job
            var parallelSortJob = new GaussianSplatBurstSorting.SortMultipleNodesJob
            {
                flattenedSplatIndices = m_FlattenedSplatIndices,
                nodeRanges = m_NodeRanges,
                allPositions = m_AllPositionsNative,
                cameraPosition = (float3)camPosition
            };

            // Calculate optimal batch size for better load balancing
            // Smaller batches = better distribution across workers
            // But too small = overhead from job scheduling
            int workerCount = Unity.Jobs.LowLevel.Unsafe.JobsUtility.JobWorkerCount;
            int batchSize = Mathf.Max(1, totalJobCount / (workerCount * 4)); // 4x oversubscription for load balancing

            // Schedule with parallel execution - DON'T complete yet
            m_SortJobHandle = parallelSortJob.Schedule(totalJobCount, batchSize);
            m_SortJobRunning = true;
            m_SortJobCameraPosition = camPosition; // Store for next frame's completion
        }

        /// <summary>
        /// Complete the pending sort job and copy sorted data back to nodes.
        /// Should be called right before we need to use the sorted data.
        /// </summary>
        void CompleteSortJob(Vector3 camPosition)
        {
            if (!m_SortJobRunning)
                return;

            // Wait for job to complete
            m_SortJobHandle.Complete();
            m_SortJobRunning = false;

            // Copy sorted data back to nodes
            for (int i = 0; i < m_NodesToSort.Length; i++)
            {
                int nodeIndex = m_NodesToSort[i];
                var node = m_Nodes[nodeIndex];
                var range = m_NodeRanges[i];

                // Copy sorted data back
                NativeArray<int>.Copy(m_FlattenedSplatIndices, range.offset, node.splatIndices.AsArray(), 0, range.length);

                // Mark as sorted
                node.isSorted = true;
                node.lastSortCameraPosition = camPosition;
            }

            if (m_SortOutliers)
            {
                var range = m_NodeRanges[m_NodesToSort.Length];
                NativeArray<int>.Copy(m_FlattenedSplatIndices, range.offset, m_OthersIndices.AsArray(), 0, range.length);

                m_OthersSorted = true;
                m_LastOthersSortCamPos = camPosition;
            }

            // Cleanup job data
            if (m_FlattenedSplatIndices.IsCreated)
                m_FlattenedSplatIndices.Dispose();
            if (m_NodeRanges.IsCreated)
                m_NodeRanges.Dispose();
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
                        if (childIndex < m_Nodes.Count)
                        {
                            var childNode = m_Nodes[childIndex];
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
