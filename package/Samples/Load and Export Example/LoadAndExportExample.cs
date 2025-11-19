// SPDX-License-Identifier: MIT

using System;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine;
using GaussianSplatting.Runtime.Utils;
using GaussianSplatting.Runtime;

namespace GaussianSplatting.Samples
{
    /// <summary>
    /// Example script that allows loading a PLY/SPZ file, importing pre-converted .gsplat files, and exporting to the optimized .gsplat format.
    /// On WebGL: Uses browser file picker for input and triggers download for output.
    /// On other platforms: Uses file system paths.
    /// </summary>
    public class LoadAndExportExample : MonoBehaviour
    {
        [Header("Desktop Settings")]
        [Tooltip("File path for desktop platforms (can be PLY or SPZ)")]
        public string inputFilePath = "path/to/your/file.ply";

        [Tooltip("File path for importing pre-converted .gsplat files")]
        public string importFilePath = "path/to/your/file.gsplat";

        [Tooltip("Output filename for the exported file")]
        public string outputFileName = "exported_gaussian_splat.gsplat";

        [Header("Conversion Settings")]
        [Tooltip("Quality setting for the conversion")]
        public GaussianSplatAssetBuilder.DataQuality quality = GaussianSplatAssetBuilder.DataQuality.Medium;

        [Header("Runtime References")]
        [Tooltip("Reference to the loaded asset (set at runtime)")]
        public GaussianSplatAsset loadedAsset;

        [Tooltip("Optional: Reference to a GaussianSplatRenderer to display the loaded asset")]
        public GaussianSplatRenderer targetRenderer;

        private bool isProcessing = false;

#if UNITY_WEBGL && !UNITY_EDITOR
        [DllImport("__Internal")]
        private static extern void PickFile(string acceptedExtensions, string gameObjectName, string callbackMethodName);

        [DllImport("__Internal")]
        private static extern void FreeFileBuffer(IntPtr ptr);
#endif

        /// <summary>
        /// Opens a file picker (WebGL) or loads from path (Desktop) and converts the file.
        /// </summary>
        public void LoadFile()
        {
            if (isProcessing)
            {
                Debug.LogWarning("Already processing a file. Please wait...");
                return;
            }

#if UNITY_WEBGL && !UNITY_EDITOR
            // On WebGL, use the browser's file picker
            PickFile(".ply,.spz", gameObject.name, nameof(OnFilePicked));
#else
            // On desktop, load from the specified path
            _ = LoadFromPathAsync(inputFilePath);
#endif
        }

        /// <summary>
        /// Exports the currently loaded asset.
        /// On WebGL: triggers a browser download.
        /// On Desktop: saves to persistent data path.
        /// </summary>
        public async void ExportAsset()
        {
            if (loadedAsset == null)
            {
                Debug.LogError("No asset loaded! Please load a file first.");
                return;
            }

            if (isProcessing)
            {
                Debug.LogWarning("Already processing. Please wait...");
                return;
            }

            try
            {
                isProcessing = true;
                Debug.Log("Exporting asset...");

#if UNITY_WEBGL && !UNITY_EDITOR
                // On WebGL, trigger browser download
                await GaussianSplatAssetRuntimeConverter.ExportAssetAsync(loadedAsset, outputFileName);
#else
                // On desktop, save to persistent data path
                string outputPath = System.IO.Path.Combine(Application.persistentDataPath, outputFileName);
                await GaussianSplatAssetRuntimeConverter.ExportAssetAsync(loadedAsset, outputPath);
                Debug.Log($"File saved to: {outputPath}");
#endif

                Debug.Log("Export completed successfully!");
            }
            catch (Exception ex)
            {
                Debug.LogError($"Export failed: {ex.Message}");
            }
            finally
            {
                isProcessing = false;
            }
        }

        /// <summary>
        /// Imports a pre-converted .gsplat file.
        /// On WebGL: Uses browser file picker for .gsplat files.
        /// On Desktop: Loads from the specified importFilePath.
        /// </summary>
        public void ImportFile()
        {
            if (isProcessing)
            {
                Debug.LogWarning("Already processing a file. Please wait...");
                return;
            }

#if UNITY_WEBGL && !UNITY_EDITOR
            // On WebGL, use the browser's file picker for .gsplat files
            PickFile(".gsplat", gameObject.name, nameof(OnGsplatFilePicked));
#else
            // On desktop, load from the specified path
            _ = ImportFromPathAsync(importFilePath);
#endif
        }

#if UNITY_WEBGL && !UNITY_EDITOR
        /// <summary>
        /// Callback for the WebGL file picker.
        /// Format: "filename|bufferPtr|bufferSize"
        /// </summary>
        private void OnFilePicked(string data)
        {
            try
            {
                if (string.IsNullOrEmpty(data))
                {
                    Debug.LogError("Callback data is null or empty");
                    return;
                }

                var parts = data.Split('|');
                if (parts.Length != 3)
                {
                    Debug.LogError($"Invalid callback data format. Expected 3 parts, got {parts.Length}");
                    return;
                }

                string fileName = parts[0];
                IntPtr bufferPtr = new IntPtr(long.Parse(parts[1]));
                int bufferSize = int.Parse(parts[2]);

                Debug.Log($"File picked: {fileName} ({bufferSize} bytes)");

                // Copy the data from the JavaScript buffer
                byte[] fileData = new byte[bufferSize];
                Marshal.Copy(bufferPtr, fileData, 0, bufferSize);

                // Free the buffer that was allocated with _malloc in JavaScript
                FreeFileBuffer(bufferPtr);

                // Process the file
                _ = ProcessFileDataAsync(fileData, fileName);
            }
            catch (Exception ex)
            {
                Debug.LogError($"Failed to process picked file: {ex.Message}");
                Debug.LogException(ex);
                isProcessing = false;
            }
        }

        /// <summary>
        /// Processes file data from the browser file picker.
        /// </summary>
        private async Task ProcessFileDataAsync(byte[] fileData, string fileName)
        {
            isProcessing = true;

            try
            {
                Debug.Log($"Processing {fileName}...");

                // Get file extension
                string extension = System.IO.Path.GetExtension(fileName);

                // Read the file data
                GaussianFileReader.ReadBytes(fileData, extension, out NativeArray<InputSplatData> inputSplats);

                if (!inputSplats.IsCreated || inputSplats.Length == 0)
                {
                    Debug.LogError("Failed to read splat data from file");
                    return;
                }

                Debug.Log($"Loaded {inputSplats.Length} splats. Converting...");

                // Convert to asset
                var settings = GaussianSplatAssetRuntimeConverter.ConversionSettings.FromQuality(quality);
                settings.progressCallback = (message, progress) =>
                {
                    Debug.Log($"[{progress:P0}] {message}");
                    return true; // Continue processing
                };

                loadedAsset = await GaussianSplatAssetRuntimeConverter.ConvertFromInputDataAsync(inputSplats, settings);
                loadedAsset.name = System.IO.Path.GetFileNameWithoutExtension(fileName);

                inputSplats.Dispose();

                // Assign to renderer if available
                if (targetRenderer != null)
                {
                    targetRenderer.m_Asset = loadedAsset;
                    Debug.Log($"Successfully loaded and converted {fileName}! Asset assigned to renderer.");
                }
                else
                {
                    Debug.Log($"Successfully loaded and converted {fileName}!");
                    Debug.LogWarning("No renderer assigned. Set 'targetRenderer' in the Inspector to display the splats.");
                }

                Debug.Log("You can now call ExportAsset() to download the optimized file.");
            }
            catch (Exception ex)
            {
                Debug.LogError($"Failed to process file: {ex.Message}");
            }
            finally
            {
                isProcessing = false;
            }
        }

        /// <summary>
        /// Callback for the WebGL file picker when importing .gsplat files.
        /// Format: "filename|bufferPtr|bufferSize"
        /// </summary>
        private void OnGsplatFilePicked(string data)
        {
            try
            {
                if (string.IsNullOrEmpty(data))
                {
                    Debug.LogError("Callback data is null or empty");
                    return;
                }

                var parts = data.Split('|');
                if (parts.Length != 3)
                {
                    Debug.LogError($"Invalid callback data format. Expected 3 parts, got {parts.Length}");
                    return;
                }

                string fileName = parts[0];
                IntPtr bufferPtr = new IntPtr(long.Parse(parts[1]));
                int bufferSize = int.Parse(parts[2]);

                Debug.Log($"File picked: {fileName} ({bufferSize} bytes)");

                // Copy the data from the JavaScript buffer
                byte[] fileData = new byte[bufferSize];
                Marshal.Copy(bufferPtr, fileData, 0, bufferSize);

                // Free the buffer that was allocated with _malloc in JavaScript
                FreeFileBuffer(bufferPtr);

                // Import the .gsplat file
                _ = ImportGsplatDataAsync(fileData, fileName);
            }
            catch (Exception ex)
            {
                Debug.LogError($"Failed to process picked file: {ex.Message}");
                Debug.LogException(ex);
                isProcessing = false;
            }
        }

        /// <summary>
        /// Imports .gsplat file data from the browser file picker.
        /// </summary>
        private async Task ImportGsplatDataAsync(byte[] fileData, string fileName)
        {
            isProcessing = true;

            try
            {
                Debug.Log($"Importing {fileName}...");

                loadedAsset = await GaussianSplatAssetRuntimeConverter.ImportAssetFromBytesAsync(fileData, System.IO.Path.GetFileNameWithoutExtension(fileName));

                // Assign to renderer if available
                if (targetRenderer != null)
                {
                    targetRenderer.m_Asset = loadedAsset;
                    Debug.Log($"Successfully imported {fileName}! Asset assigned to renderer.");
                }
                else
                {
                    Debug.Log($"Successfully imported {fileName}!");
                    Debug.LogWarning("No renderer assigned. Set 'targetRenderer' in the Inspector to display the splats.");
                }

                Debug.Log($"Loaded {loadedAsset.splatCount:N0} splats.");
            }
            catch (Exception ex)
            {
                Debug.LogError($"Failed to import file: {ex.Message}");
            }
            finally
            {
                isProcessing = false;
            }
        }
#endif

        /// <summary>
        /// Loads a file from a path (Desktop platforms).
        /// </summary>
        private async Task LoadFromPathAsync(string filePath)
        {
            isProcessing = true;

            try
            {
                Debug.Log($"Loading file: {filePath}");

                var settings = GaussianSplatAssetRuntimeConverter.ConversionSettings.FromQuality(quality);
                settings.progressCallback = (message, progress) =>
                {
                    Debug.Log($"[{progress:P0}] {message}");
                    return true; // Continue processing
                };

                loadedAsset = await GaussianSplatAssetRuntimeConverter.ConvertFromFileAsync(filePath, settings);

                // Assign to renderer if available
                if (targetRenderer != null)
                {
                    targetRenderer.m_Asset = loadedAsset;
                    Debug.Log($"Successfully loaded and converted {filePath}! Asset assigned to renderer.");
                }
                else
                {
                    Debug.Log($"Successfully loaded and converted {filePath}!");
                    Debug.LogWarning("No renderer assigned. Set 'targetRenderer' in the Inspector to display the splats.");
                }

                Debug.Log("You can now call ExportAsset() to save the optimized file.");
            }
            catch (Exception ex)
            {
                Debug.LogError($"Failed to load file: {ex.Message}");
            }
            finally
            {
                isProcessing = false;
            }
        }

        /// <summary>
        /// Imports a .gsplat file from a path (Desktop platforms).
        /// </summary>
        private async Task ImportFromPathAsync(string filePath)
        {
            isProcessing = true;

            try
            {
                Debug.Log($"Importing file: {filePath}");

                loadedAsset = await GaussianSplatAssetRuntimeConverter.ImportAssetAsync(filePath);

                // Assign to renderer if available
                if (targetRenderer != null)
                {
                    targetRenderer.m_Asset = loadedAsset;
                    Debug.Log($"Successfully imported {filePath}! Asset assigned to renderer.");
                }
                else
                {
                    Debug.Log($"Successfully imported {filePath}!");
                    Debug.LogWarning("No renderer assigned. Set 'targetRenderer' in the Inspector to display the splats.");
                }

                Debug.Log($"Loaded {loadedAsset.splatCount:N0} splats.");
            }
            catch (Exception ex)
            {
                Debug.LogError($"Failed to import file: {ex.Message}");
            }
            finally
            {
                isProcessing = false;
            }
        }

        /// <summary>
        /// Example: Press L to load, E to export, I to import
        /// </summary>
        private void Update()
        {
            if (Input.GetKeyDown(KeyCode.L))
            {
                LoadFile();
            }

            if (Input.GetKeyDown(KeyCode.E))
            {
                ExportAsset();
            }

            if (Input.GetKeyDown(KeyCode.I))
            {
                ImportFile();
            }
        }

        private void OnGUI()
        {
            GUILayout.BeginArea(new Rect(10, 10, 300, 250));
            GUILayout.Label("Gaussian Splat Load & Export Example");
            GUILayout.Space(10);

            if (GUILayout.Button("Load File (L)") && !isProcessing)
            {
                LoadFile();
            }

            if (GUILayout.Button("Import .gsplat (I)") && !isProcessing)
            {
                ImportFile();
            }

            GUI.enabled = loadedAsset != null && !isProcessing;
            if (GUILayout.Button("Export Asset (E)"))
            {
                ExportAsset();
            }
            GUI.enabled = true;

            GUILayout.Space(10);

            if (loadedAsset != null)
            {
                GUILayout.Label($"Loaded: {loadedAsset.name}");
                GUILayout.Label($"Splats: {loadedAsset.splatCount:N0}");
            }
            else
            {
                GUILayout.Label("No asset loaded");
            }

            if (isProcessing)
            {
                GUILayout.Label("Processing...");
            }

            GUILayout.EndArea();
        }
    }
}
