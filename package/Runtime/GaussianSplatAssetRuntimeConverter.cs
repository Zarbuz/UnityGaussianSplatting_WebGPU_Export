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
        /// </summary>
        /// <param name="filePath">Path to the PLY/SPZ file</param>
        /// <param name="settings">Conversion settings (optional, uses default if null)</param>
        /// <returns>A GaussianSplatAsset ready to use at runtime</returns>
        public static async Task<GaussianSplatAsset> ConvertFromFileAsync(string filePath, ConversionSettings? settings = null)
        {
            var conversionSettings = settings ?? ConversionSettings.Default;

            try
            {
                conversionSettings.progressCallback?.Invoke("Loading file", 0.0f);

                if (!File.Exists(filePath))
                    throw new FileNotFoundException($"File not found: {filePath}");

                // Read the input file
                conversionSettings.progressCallback?.Invoke("Reading input file", 0.05f);
                GaussianFileReader.ReadFile(filePath, out NativeArray<InputSplatData> inputSplats);

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

                // Write to a temporary file to use GaussianFileReader
                string tempPath = Path.Combine(Application.temporaryCachePath, $"temp_splat_{Guid.NewGuid()}.ply");
                try
                {
                    await File.WriteAllBytesAsync(tempPath, fileData);
                    var asset = await ConvertFromFileAsync(tempPath, conversionSettings);

                    // Set the asset name from the URL
                    string fileName = Path.GetFileNameWithoutExtension(new Uri(url).LocalPath);
                    asset.name = fileName;

                    return asset;
                }
                finally
                {
                    // Clean up temp file
                    if (File.Exists(tempPath))
                        File.Delete(tempPath);
                }
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

            try
            {
                // Wrap the progress callback to offset by 30% (file loading is 0-30%)
                bool WrappedProgress(string message, float progress)
                {
                    return conversionSettings.progressCallback?.Invoke(message, 0.3f + progress * 0.7f) ?? true;
                }

                // Use the builder to convert the data
                var builder = new GaussianSplatAssetBuilder(conversionSettings.buildSettings, WrappedProgress);

                // Run the build on a background thread to avoid blocking
                var buildResult = await Task.Run(() => builder.BuildAsset(inputSplats));

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

                // Set the runtime data (convert NativeArray to byte[])
                asset.SetRuntimeData(
                    buildResult.chunkData.IsCreated ? buildResult.chunkData.ToArray() : null,
                    buildResult.posData.ToArray(),
                    buildResult.otherData.ToArray(),
                    buildResult.colorData.ToArray(),
                    buildResult.shData.ToArray()
                );

                // Clean up build result
                buildResult.Dispose();

                WrappedProgress("Conversion complete", 1.0f);

                return asset;
            }
            catch (Exception ex)
            {
                conversionSettings.progressCallback?.Invoke($"Error during conversion: {ex.Message}", 1.0f);
                throw;
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
    }
}
