// SPDX-License-Identifier: MIT

using System;
using System.Threading.Tasks;
using UnityEngine;

namespace GaussianSplatting.Runtime
{
    /// <summary>
    /// Example component that demonstrates how to load Gaussian Splat assets at runtime.
    /// Attach this to a GameObject with a GaussianSplatRenderer component.
    /// </summary>
    [RequireComponent(typeof(GaussianSplatRenderer))]
    public class GaussianSplatRuntimeLoader : MonoBehaviour
    {
        [Header("Source")]
        [Tooltip("Load from a file path or URL")]
        public bool loadFromUrl = false;

        [Tooltip("File path (local) or URL to load the PLY/SPZ file from")]
        public string sourcePathOrUrl = "";

        [Tooltip("Load automatically on Start")]
        public bool loadOnStart = true;

        [Header("Quality Settings")]
        [Tooltip("Quality preset for the conversion")]
        public GaussianSplatAssetBuilder.DataQuality quality = GaussianSplatAssetBuilder.DataQuality.Medium;

        [Header("Status")]
        [SerializeField] private bool isLoading = false;
        [SerializeField] private string loadingStatus = "";
        [SerializeField] private float loadingProgress = 0f;

        private GaussianSplatRenderer m_Renderer;

        public bool IsLoading => isLoading;
        public string LoadingStatus => loadingStatus;
        public float LoadingProgress => loadingProgress;

        void Awake()
        {
            m_Renderer = GetComponent<GaussianSplatRenderer>();
        }

        void Start()
        {
            if (loadOnStart && !string.IsNullOrEmpty(sourcePathOrUrl))
            {
                _ = LoadSplatAsync();
            }
        }

        /// <summary>
        /// Loads a Gaussian Splat from the configured source (file or URL).
        /// </summary>
        public async Task<bool> LoadSplatAsync()
        {
            if (isLoading)
            {
                Debug.LogWarning("Already loading a splat asset");
                return false;
            }

            if (string.IsNullOrEmpty(sourcePathOrUrl))
            {
                Debug.LogError("No source path or URL specified");
                return false;
            }

            isLoading = true;
            loadingProgress = 0f;
            loadingStatus = "Starting...";

            try
            {
                var settings = GaussianSplatAssetRuntimeConverter.ConversionSettings.FromQuality(quality);
                settings.progressCallback = OnProgress;

                GaussianSplatAsset asset;

                // Auto-detect: In WebGL, StreamingAssets must be loaded via URL
                bool shouldUseUrl = loadFromUrl;
#if UNITY_WEBGL && !UNITY_EDITOR
                if (sourcePathOrUrl.StartsWith("StreamingAssets", StringComparison.OrdinalIgnoreCase))
                {
                    shouldUseUrl = true;
                }
#endif

                if (shouldUseUrl)
                {
                    string url = sourcePathOrUrl;

                    // Convert StreamingAssets path to URL for WebGL
#if UNITY_WEBGL && !UNITY_EDITOR
                    if (sourcePathOrUrl.StartsWith("StreamingAssets/", StringComparison.OrdinalIgnoreCase) ||
                        sourcePathOrUrl.StartsWith("StreamingAssets\\", StringComparison.OrdinalIgnoreCase))
                    {
                        // Extract relative path after "StreamingAssets/"
                        string relativePath = sourcePathOrUrl.Substring("StreamingAssets/".Length);
                        url = Application.streamingAssetsPath + "/" + relativePath;
                    }
                    else if (sourcePathOrUrl.Equals("StreamingAssets", StringComparison.OrdinalIgnoreCase))
                    {
                        url = Application.streamingAssetsPath;
                    }
#endif

                    Debug.Log($"Loading Gaussian Splat from URL: {url}");
                    asset = await GaussianSplatAssetRuntimeConverter.ConvertFromUrlAsync(url, settings);
                }
                else
                {
                    Debug.Log($"Loading Gaussian Splat from file: {sourcePathOrUrl}");
                    asset = await GaussianSplatAssetRuntimeConverter.ConvertFromFileAsync(sourcePathOrUrl, settings);
                }

                if (asset != null)
                {
                    // Assign the asset to the renderer
                    m_Renderer.m_Asset = asset;

                    // The renderer's Update() method will automatically detect the asset change
                    // and recreate GPU resources on the next frame. This is safer than toggling
                    // the enabled state, which could interrupt ongoing async operations like octree building.

                    loadingStatus = "Complete!";
                    loadingProgress = 1f;
                    Debug.Log($"Successfully loaded Gaussian Splat: {asset.name} ({asset.splatCount} splats)");
                    return true;
                }
                else
                {
                    loadingStatus = "Failed: Asset is null";
                    Debug.LogError("Failed to load Gaussian Splat: Asset is null");
                    return false;
                }
            }
            catch (Exception ex)
            {
                loadingStatus = $"Error: {ex.Message}";
                Debug.LogError($"Failed to load Gaussian Splat: {ex.Message}\n{ex.StackTrace}");
                return false;
            }
            finally
            {
                isLoading = false;
            }
        }

        /// <summary>
        /// Loads a Gaussian Splat from a specific path or URL.
        /// </summary>
        public async Task<bool> LoadSplatFromSourceAsync(string pathOrUrl, bool isUrl)
        {
            sourcePathOrUrl = pathOrUrl;
            loadFromUrl = isUrl;
            return await LoadSplatAsync();
        }

        private bool OnProgress(string message, float progress)
        {
			Debug.Log("[GaussianSplatRuntimeLoader] " + message + " " + progress);
            loadingStatus = message;
            loadingProgress = progress;
            return true; // Return false to cancel
        }

        void OnValidate()
        {
            // Ensure we have a renderer
            if (m_Renderer == null)
                m_Renderer = GetComponent<GaussianSplatRenderer>();
        }
    }
}
