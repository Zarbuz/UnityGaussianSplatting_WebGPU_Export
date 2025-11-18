// SPDX-License-Identifier: MIT

using System;
using System.IO;
using System.Threading.Tasks;
using Unity.Collections;
using UnityEngine;
using UnityEngine.Networking;
using GaussianSplatting.Runtime.Utils;

namespace GaussianSplatting.Runtime
{
    /// <summary>
    /// Runtime converter for loading and converting Gaussian Splat files (PLY/SPZ) at runtime.
    /// Supports loading from file paths and URLs, and converts them to GaussianSplatAsset instances.
    /// </summary>
    public class GaussianSplatAssetRuntimeConverter
    {
        /// <summary>
        /// Gets the recommended quality setting based on splat count for WebGL builds.
        /// WebGL has memory constraints, so we need to use lower quality for large files.
        /// </summary>
        public static GaussianSplatAssetBuilder.DataQuality GetRecommendedQualityForWebGL(int splatCount)
        {
#if UNITY_WEBGL && !UNITY_EDITOR
            // WebGL memory-optimized recommendations
            if (splatCount < 100_000)
                return GaussianSplatAssetBuilder.DataQuality.High;
            else if (splatCount < 500_000)
                return GaussianSplatAssetBuilder.DataQuality.Medium;
            else if (splatCount < 1_000_000)
                return GaussianSplatAssetBuilder.DataQuality.Low;
            else
                return GaussianSplatAssetBuilder.DataQuality.VeryLow;
#else
            // Desktop - can handle higher quality
            if (splatCount < 500_000)
                return GaussianSplatAssetBuilder.DataQuality.High;
            else if (splatCount < 2_000_000)
                return GaussianSplatAssetBuilder.DataQuality.Medium;
            else
                return GaussianSplatAssetBuilder.DataQuality.Low;
#endif
        }
        /// <summary>
        /// Progress callback for reporting conversion progress.
        /// Parameters: (message, progress 0-1)
        /// Return false to cancel the operation.
        /// </summary>
        public delegate bool ProgressCallback(string message, float progress);

        /// <summary>
        /// Settings for runtime conversion.
        /// </summary>
        public struct ConversionSettings
        {
            public GaussianSplatAssetBuilder.BuildSettings buildSettings;
            public ProgressCallback progressCallback;

            public static ConversionSettings Default => new ConversionSettings
            {
                buildSettings = GaussianSplatAssetBuilder.BuildSettings.Default,
                progressCallback = null
            };

            public static ConversionSettings FromQuality(GaussianSplatAssetBuilder.DataQuality quality)
            {
                return new ConversionSettings
                {
                    buildSettings = GaussianSplatAssetBuilder.BuildSettings.FromQuality(quality),
                    progressCallback = null
                };
            }
        }

        /// <summary>
        /// Loads a PLY/SPZ file from a file path and converts it to a GaussianSplatAsset at runtime.
        /// Supports absolute paths, relative paths, and StreamingAssets paths.
        /// </summary>
        /// <param name="filePath">Path to the PLY/SPZ file. Can be:
        /// - Absolute path: "C:/MyFiles/splat.ply"
        /// - Relative path: "Data/splat.ply"
        /// - StreamingAssets path: "Assets/StreamingAssets/splat.ply" or "StreamingAssets/splat.ply"
        /// </param>
        /// <param name="settings">Conversion settings (optional, uses default if null)</param>
        /// <returns>A GaussianSplatAsset ready to use at runtime</returns>
        public static async Task<GaussianSplatAsset> ConvertFromFileAsync(string filePath, ConversionSettings? settings = null)
        {
            var conversionSettings = settings ?? ConversionSettings.Default;

            try
            {
                conversionSettings.progressCallback?.Invoke("Loading file", 0.0f);

                // Resolve the file path (handles StreamingAssets, relative, and absolute paths)
                string resolvedPath = ResolveFilePath(filePath);

                if (!File.Exists(resolvedPath))
                    throw new FileNotFoundException($"File not found: {resolvedPath} (original path: {filePath})");

                // Read the input file
                conversionSettings.progressCallback?.Invoke("Reading input file", 0.05f);
                GaussianFileReader.ReadFile(resolvedPath, out NativeArray<InputSplatData> inputSplats);

                if (!inputSplats.IsCreated || inputSplats.Length == 0)
                    throw new InvalidDataException("Failed to read splat data from file");

                // Convert using the builder
                var asset = await ConvertFromInputDataAsync(inputSplats, conversionSettings);

                // Set the asset name from the file
                asset.name = Path.GetFileNameWithoutExtension(filePath);

                inputSplats.Dispose();
                return asset;
            }
            catch (Exception ex)
            {
                conversionSettings.progressCallback?.Invoke($"Error: {ex.Message}", 1.0f);
                throw;
            }
        }

        /// <summary>
        /// Downloads a PLY/SPZ file from a URL and converts it to a GaussianSplatAsset at runtime.
        /// </summary>
        /// <param name="url">URL to download the PLY/SPZ file from</param>
        /// <param name="settings">Conversion settings (optional, uses default if null)</param>
        /// <returns>A GaussianSplatAsset ready to use at runtime</returns>
        public static async Task<GaussianSplatAsset> ConvertFromUrlAsync(string url, ConversionSettings? settings = null)
        {
            var conversionSettings = settings ?? ConversionSettings.Default;

            try
            {
                conversionSettings.progressCallback?.Invoke("Downloading file", 0.0f);

                // Download the file
                byte[] fileData = await DownloadFileAsync(url, (progress) =>
                {
                    conversionSettings.progressCallback?.Invoke($"Downloading: {progress:P0}", progress * 0.3f);
                });

                if (fileData == null || fileData.Length == 0)
                    throw new InvalidDataException("Failed to download file or file is empty");

                conversionSettings.progressCallback?.Invoke("Processing downloaded file", 0.3f);

                // Read directly from bytes - efficient on all platforms
                string fileExtension = Path.GetExtension(new Uri(url).LocalPath);
                conversionSettings.progressCallback?.Invoke("Reading input data", 0.05f);
                GaussianFileReader.ReadBytes(fileData, fileExtension, out NativeArray<InputSplatData> inputSplats);

                if (!inputSplats.IsCreated || inputSplats.Length == 0)
                    throw new InvalidDataException("Failed to read splat data from downloaded file");

                // Convert using the builder
                var asset = await ConvertFromInputDataAsync(inputSplats, conversionSettings);

                // Set the asset name from the URL
                string fileName = Path.GetFileNameWithoutExtension(new Uri(url).LocalPath);
                asset.name = fileName;

                inputSplats.Dispose();
                return asset;
            }
            catch (Exception ex)
            {
                conversionSettings.progressCallback?.Invoke($"Error: {ex.Message}", 1.0f);
                throw;
            }
        }

        /// <summary>
        /// Converts InputSplatData directly to a GaussianSplatAsset at runtime.
        /// Useful if you already have the splat data loaded in memory.
        /// </summary>
        /// <param name="inputSplats">Input splat data</param>
        /// <param name="settings">Conversion settings (optional, uses default if null)</param>
        /// <returns>A GaussianSplatAsset ready to use at runtime</returns>
        public static async Task<GaussianSplatAsset> ConvertFromInputDataAsync(NativeArray<InputSplatData> inputSplats, ConversionSettings? settings = null)
        {
            var conversionSettings = settings ?? ConversionSettings.Default;

            if (!inputSplats.IsCreated || inputSplats.Length == 0)
                throw new ArgumentException("Input splats array is empty or not created");

            // Build result to be disposed in finally block if error occurs
            GaussianSplatAssetBuilder.BuildResult buildResult = default;
            bool ownershipTransferred = false;

            try
            {
                // Wrap the progress callback to offset by 30% (file loading is 0-30%)
                bool WrappedProgress(string message, float progress)
                {
                    return conversionSettings.progressCallback?.Invoke(message, 0.3f + progress * 0.7f) ?? true;
                }

                // Use the builder to convert the data
                var builder = new GaussianSplatAssetBuilder(conversionSettings.buildSettings, WrappedProgress);

                // BuildAsset is now async with yields, so just await it directly
                buildResult = await builder.BuildAsset(inputSplats);

                WrappedProgress("Creating asset", 0.95f);

                // Create the asset and populate with runtime data
                var asset = ScriptableObject.CreateInstance<GaussianSplatAsset>();
                asset.Initialize(
                    inputSplats.Length,
                    conversionSettings.buildSettings.formatPos,
                    conversionSettings.buildSettings.formatScale,
                    conversionSettings.buildSettings.formatColor,
                    conversionSettings.buildSettings.formatSH,
                    buildResult.boundsMin,
                    buildResult.boundsMax,
                    null // No camera info for runtime conversion
                );
                asset.SetDataHash(buildResult.dataHash);

                // Set the runtime data directly - no conversion needed!
                // Asset takes ownership of the NativeArrays, so we don't dispose them
                asset.SetRuntimeData(
                    buildResult.chunkData,
                    buildResult.posData,
                    buildResult.otherData,
                    buildResult.colorData,
                    buildResult.shData
                );

                // Mark ownership as transferred - asset will dispose the arrays
                ownershipTransferred = true;

                // Just yield to allow other work
                await Task.Yield();

                WrappedProgress("Conversion complete", 1.0f);

                return asset;
            }
            catch (Exception ex)
            {
                conversionSettings.progressCallback?.Invoke($"Error during conversion: {ex.Message}", 1.0f);
                throw;
            }
            finally
            {
                // Only dispose if ownership was not transferred (i.e., error occurred before SetRuntimeData)
                if (!ownershipTransferred && buildResult.posData.IsCreated)
                    buildResult.Dispose();
            }
        }

        /// <summary>
        /// Downloads a file from a URL asynchronously.
        /// </summary>
        private static async Task<byte[]> DownloadFileAsync(string url, Action<float> progressCallback = null)
        {
            using (UnityWebRequest request = UnityWebRequest.Get(url))
            {
                var operation = request.SendWebRequest();

                while (!operation.isDone)
                {
                    progressCallback?.Invoke(operation.progress);
                    await Task.Yield();
                }

                if (request.result != UnityWebRequest.Result.Success)
                {
                    throw new Exception($"Failed to download file: {request.error}");
                }

                progressCallback?.Invoke(1.0f);
                return request.downloadHandler.data;
            }
        }

        /// <summary>
        /// Resolves a file path to an absolute path, handling StreamingAssets paths correctly.
        /// </summary>
        /// <param name="filePath">The input file path (can be absolute, relative, or StreamingAssets path)</param>
        /// <returns>Resolved absolute file path</returns>
        private static string ResolveFilePath(string filePath)
        {
            if (string.IsNullOrEmpty(filePath))
                return filePath;

            // Check if it's already an absolute path
            if (Path.IsPathRooted(filePath))
            {
                // Check if it's an Editor-style StreamingAssets path (starts with "Assets/StreamingAssets")
                if (filePath.StartsWith("Assets/StreamingAssets/", StringComparison.OrdinalIgnoreCase) ||
                    filePath.StartsWith("Assets\\StreamingAssets\\", StringComparison.OrdinalIgnoreCase))
                {
                    // Extract the relative path after "Assets/StreamingAssets/"
                    string relativePath = filePath.Substring("Assets/StreamingAssets/".Length);
                    return Path.Combine(Application.streamingAssetsPath, relativePath);
                }

                // It's a regular absolute path, return as-is
                return filePath;
            }

            // Check if it starts with "StreamingAssets/" (without "Assets/")
            if (filePath.StartsWith("StreamingAssets/", StringComparison.OrdinalIgnoreCase) ||
                filePath.StartsWith("StreamingAssets\\", StringComparison.OrdinalIgnoreCase))
            {
                // Extract the relative path after "StreamingAssets/"
                string relativePath = filePath.Substring("StreamingAssets/".Length);
                return Path.Combine(Application.streamingAssetsPath, relativePath);
            }

            // Otherwise, treat it as a relative path from StreamingAssets
            // This allows users to just specify the filename if it's in StreamingAssets root
            string streamingPath = Path.Combine(Application.streamingAssetsPath, filePath);
            if (File.Exists(streamingPath))
                return streamingPath;

            // If not found in StreamingAssets, treat as relative to current directory
            // or return the original path and let the caller handle the error
            return Path.GetFullPath(filePath);
        }
    }
}
