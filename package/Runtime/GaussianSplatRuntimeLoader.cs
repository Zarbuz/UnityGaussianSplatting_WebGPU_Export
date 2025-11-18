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

        [Tooltip("File path (local) or URL to load from. Supports PLY, SPZ, and .gsplat (internal format)")]
        public string sourcePathOrUrl = "";

        [Tooltip("Load automatically on Start")]
        public bool loadOnStart = true;

        [Header("Format Options")]
        [Tooltip("Use internal .gsplat format (fast) or convert from PLY/SPZ (slower but universal)")]
        public bool useInternalFormat = false;

        [Tooltip("If converting from PLY/SPZ, export to .gsplat for faster subsequent loads")]
        public bool exportAfterConversion = false;

        [Tooltip("Path to export .gsplat file (only used if exportAfterConversion is true). Leave empty to auto-generate.")]
        public string exportPath = "";

        [Header("Quality Settings (PLY/SPZ conversion only)")]
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
                GaussianSplatAsset asset;

                // Auto-detect: In WebGL, StreamingAssets must be loaded via URL
                bool shouldUseUrl = loadFromUrl;
#if UNITY_WEBGL && !UNITY_EDITOR
                if (sourcePathOrUrl.StartsWith("StreamingAssets", StringComparison.OrdinalIgnoreCase))
                {
                    shouldUseUrl = true;
                }
#endif

                // Resolve path/URL for StreamingAssets
                string resolvedSource = sourcePathOrUrl;
                if (sourcePathOrUrl.StartsWith("StreamingAssets/", StringComparison.OrdinalIgnoreCase))
                {
                    // Extract relative path after "StreamingAssets/"
                    string relativePath = sourcePathOrUrl.Substring("StreamingAssets/".Length);
                    resolvedSource = Application.streamingAssetsPath + "/" + relativePath;
                }

                // Check if using internal .gsplat format
                if (useInternalFormat)
                {
                    loadingStatus = "Loading internal format...";
                    Debug.Log($"Loading .gsplat from: {resolvedSource}");

                    // For URLs (including WebGL StreamingAssets), download and import from bytes
                    if (shouldUseUrl)
                    {
                        loadingStatus = "Downloading .gsplat file...";
                        byte[] fileData = await DownloadFileAsync(resolvedSource);

                        loadingStatus = "Importing .gsplat data...";
                        string assetName = System.IO.Path.GetFileNameWithoutExtension(sourcePathOrUrl);
                        asset = await GaussianSplatAssetRuntimeConverter.ImportAssetFromBytesAsync(fileData, assetName);
                    }
                    else
                    {
                        // Local file - can import directly
                        asset = await GaussianSplatAssetRuntimeConverter.ImportAssetAsync(resolvedSource);
                    }
                }
                else
                {
                    // Converting from PLY/SPZ
                    var settings = GaussianSplatAssetRuntimeConverter.ConversionSettings.FromQuality(quality);
                    settings.progressCallback = OnProgress;

                    if (shouldUseUrl)
                    {
                        Debug.Log($"Converting Gaussian Splat from URL: {resolvedSource}");
                        asset = await GaussianSplatAssetRuntimeConverter.ConvertFromUrlAsync(resolvedSource, settings);
                    }
                    else
                    {
                        Debug.Log($"Converting Gaussian Splat from file: {resolvedSource}");
                        asset = await GaussianSplatAssetRuntimeConverter.ConvertFromFileAsync(resolvedSource, settings);
                    }

                    // Export to .gsplat if requested
                    if (asset != null && exportAfterConversion)
                    {
                        string outputPath = exportPath;
                        if (string.IsNullOrEmpty(outputPath))
                        {
                            // Auto-generate export path in PersistentDataPath
                            string filename = System.IO.Path.GetFileNameWithoutExtension(sourcePathOrUrl);
                            outputPath = System.IO.Path.Combine(Application.persistentDataPath, $"{filename}.gsplat");
                        }

                        loadingStatus = "Exporting internal format...";
                        Debug.Log($"Exporting to .gsplat: {outputPath}");
                        await GaussianSplatAssetRuntimeConverter.ExportAssetAsync(asset, outputPath);
                    }
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

        /// <summary>
        /// Downloads a file from a URL asynchronously.
        /// </summary>
        private async Task<byte[]> DownloadFileAsync(string url)
        {
            using (UnityEngine.Networking.UnityWebRequest request = UnityEngine.Networking.UnityWebRequest.Get(url))
            {
                var operation = request.SendWebRequest();

                while (!operation.isDone)
                {
                    loadingProgress = operation.progress * 0.3f; // Use first 30% for download
                    await Task.Yield();
                }

                if (request.result != UnityEngine.Networking.UnityWebRequest.Result.Success)
                {
                    throw new Exception($"Failed to download file from {url}: {request.error}");
                }

                loadingProgress = 0.3f;
                return request.downloadHandler.data;
            }
        }

        void OnValidate()
        {
            // Ensure we have a renderer
            if (m_Renderer == null)
                m_Renderer = GetComponent<GaussianSplatRenderer>();
        }
    }
}
