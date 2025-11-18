// SPDX-License-Identifier: MIT

using GaussianSplatting.Runtime;
using GaussianSplatting.Runtime.Utils;
using System;
using System.Collections.Generic;
using System.IO;
using GaussianSplatting.Editor.Utils;
using Unity.Burst;
using Unity.Collections;
using UnityEditor;
using UnityEngine;

namespace GaussianSplatting.Editor
{
	[BurstCompile]
	public class GaussianSplatAssetCreator : EditorWindow
	{
		const string kProgressTitle = "Creating Gaussian Splat Asset";
		const string kCamerasJson = "cameras.json";
		const string kPrefQuality = "nesnausk.GaussianSplatting.CreatorQuality";
		const string kPrefOutputFolder = "nesnausk.GaussianSplatting.CreatorOutputFolder";

		enum DataQuality
		{
			VeryHigh,
			High,
			Medium,
			Low,
			VeryLow,
			Custom,
		}

		readonly FilePickerControl m_FilePicker = new();

		[SerializeField] string m_InputFile;
		[SerializeField] bool m_ImportCameras = true;

		[SerializeField] string m_OutputFolder = "Assets/GaussianAssets";
		[SerializeField] DataQuality m_Quality = DataQuality.Medium;
		[SerializeField] GaussianSplatAsset.VectorFormat m_FormatPos;
		[SerializeField] GaussianSplatAsset.VectorFormat m_FormatScale;
		[SerializeField] GaussianSplatAsset.ColorFormat m_FormatColor;
		[SerializeField] GaussianSplatAsset.SHFormat m_FormatSH;

		string m_ErrorMessage;
		string m_PrevFilePath;
		int m_PrevVertexCount;
		long m_PrevFileSize;

		bool isUsingChunks =>
			m_FormatPos != GaussianSplatAsset.VectorFormat.Float32 ||
			m_FormatScale != GaussianSplatAsset.VectorFormat.Float32 ||
			m_FormatColor != GaussianSplatAsset.ColorFormat.Float32x4 ||
			m_FormatSH != GaussianSplatAsset.SHFormat.Float32;

		[MenuItem("Tools/Gaussian Splats/Create GaussianSplatAsset")]
		public static void Init()
		{
			var window = GetWindowWithRect<GaussianSplatAssetCreator>(new Rect(50, 50, 360, 340), false, "Gaussian Splat Creator", true);
			window.minSize = new Vector2(320, 320);
			window.maxSize = new Vector2(1500, 1500);
			window.Show();
		}

		void Awake()
		{
			m_Quality = (DataQuality)EditorPrefs.GetInt(kPrefQuality, (int)DataQuality.Medium);
			m_OutputFolder = EditorPrefs.GetString(kPrefOutputFolder, "Assets/GaussianAssets");
		}

		void OnEnable()
		{
			ApplyQualityLevel();
		}

		void OnGUI()
		{
			EditorGUILayout.Space();
			GUILayout.Label("Input data", EditorStyles.boldLabel);
			var rect = EditorGUILayout.GetControlRect(true);
			m_InputFile = m_FilePicker.PathFieldGUI(rect, new GUIContent("Input PLY/SPZ File"), m_InputFile, "ply,spz", "PointCloudFile");
			m_ImportCameras = EditorGUILayout.Toggle("Import Cameras", m_ImportCameras);

			if (m_InputFile != m_PrevFilePath && !string.IsNullOrWhiteSpace(m_InputFile))
			{
				m_PrevVertexCount = 0;
				m_ErrorMessage = null;
				try
				{
					m_PrevVertexCount = GaussianFileReader.ReadFileHeader(m_InputFile);
				}
				catch (Exception ex)
				{
					m_ErrorMessage = ex.Message;
				}

				m_PrevFileSize = File.Exists(m_InputFile) ? new FileInfo(m_InputFile).Length : 0;
				m_PrevFilePath = m_InputFile;
			}

			if (m_PrevVertexCount > 0)
				EditorGUILayout.LabelField("File Size", $"{EditorUtility.FormatBytes(m_PrevFileSize)} - {m_PrevVertexCount:N0} splats");
			else
				GUILayout.Space(EditorGUIUtility.singleLineHeight);

			EditorGUILayout.Space();
			GUILayout.Label("Output", EditorStyles.boldLabel);
			rect = EditorGUILayout.GetControlRect(true);
			string newOutputFolder = m_FilePicker.PathFieldGUI(rect, new GUIContent("Output Folder"), m_OutputFolder, null, "GaussianAssetOutputFolder");
			if (newOutputFolder != m_OutputFolder)
			{
				m_OutputFolder = newOutputFolder;
				EditorPrefs.SetString(kPrefOutputFolder, m_OutputFolder);
			}

			var newQuality = (DataQuality)EditorGUILayout.EnumPopup("Quality", m_Quality);
			if (newQuality != m_Quality)
			{
				m_Quality = newQuality;
				EditorPrefs.SetInt(kPrefQuality, (int)m_Quality);
				ApplyQualityLevel();
			}

			long sizePos = 0, sizeOther = 0, sizeCol = 0, sizeSHs = 0, totalSize = 0;
			if (m_PrevVertexCount > 0)
			{
				sizePos = GaussianSplatAsset.CalcPosDataSize(m_PrevVertexCount, m_FormatPos);
				sizeOther = GaussianSplatAsset.CalcOtherDataSize(m_PrevVertexCount, m_FormatScale);
				sizeCol = GaussianSplatAsset.CalcColorDataSize(m_PrevVertexCount, m_FormatColor);
				sizeSHs = GaussianSplatAsset.CalcSHDataSize(m_PrevVertexCount, m_FormatSH);
				long sizeChunk = isUsingChunks ? GaussianSplatAsset.CalcChunkDataSize(m_PrevVertexCount) : 0;
				totalSize = sizePos + sizeOther + sizeCol + sizeSHs + sizeChunk;
			}

			const float kSizeColWidth = 70;
			EditorGUI.BeginDisabledGroup(m_Quality != DataQuality.Custom);
			EditorGUI.indentLevel++;
			GUILayout.BeginHorizontal();
			m_FormatPos = (GaussianSplatAsset.VectorFormat)EditorGUILayout.EnumPopup("Position", m_FormatPos);
			GUILayout.Label(sizePos > 0 ? EditorUtility.FormatBytes(sizePos) : string.Empty, GUILayout.Width(kSizeColWidth));
			GUILayout.EndHorizontal();
			GUILayout.BeginHorizontal();
			m_FormatScale = (GaussianSplatAsset.VectorFormat)EditorGUILayout.EnumPopup("Scale", m_FormatScale);
			GUILayout.Label(sizeOther > 0 ? EditorUtility.FormatBytes(sizeOther) : string.Empty, GUILayout.Width(kSizeColWidth));
			GUILayout.EndHorizontal();
			GUILayout.BeginHorizontal();
			m_FormatColor = (GaussianSplatAsset.ColorFormat)EditorGUILayout.EnumPopup("Color", m_FormatColor);
			GUILayout.Label(sizeCol > 0 ? EditorUtility.FormatBytes(sizeCol) : string.Empty, GUILayout.Width(kSizeColWidth));
			GUILayout.EndHorizontal();
			GUILayout.BeginHorizontal();
			m_FormatSH = (GaussianSplatAsset.SHFormat)EditorGUILayout.EnumPopup("SH", m_FormatSH);
			GUIContent shGC = new GUIContent();
			shGC.text = sizeSHs > 0 ? EditorUtility.FormatBytes(sizeSHs) : string.Empty;
			if (m_FormatSH >= GaussianSplatAsset.SHFormat.Cluster64k)
			{
				shGC.tooltip = "Note that SH clustering is not fast! (3-10 minutes for 6M splats)";
				shGC.image = EditorGUIUtility.IconContent("console.warnicon.sml").image;
			}
			GUILayout.Label(shGC, GUILayout.Width(kSizeColWidth));
			GUILayout.EndHorizontal();
			EditorGUI.indentLevel--;
			EditorGUI.EndDisabledGroup();
			if (totalSize > 0)
				EditorGUILayout.LabelField("Asset Size", $"{EditorUtility.FormatBytes(totalSize)} - {(double)m_PrevFileSize / totalSize:F2}x smaller");
			else
				GUILayout.Space(EditorGUIUtility.singleLineHeight);


			EditorGUILayout.Space();
			GUILayout.BeginHorizontal();
			GUILayout.Space(30);
			if (GUILayout.Button("Create Asset"))
			{
				CreateAsset();
			}
			GUILayout.Space(30);
			GUILayout.EndHorizontal();

			if (!string.IsNullOrWhiteSpace(m_ErrorMessage))
			{
				EditorGUILayout.HelpBox(m_ErrorMessage, MessageType.Error);
			}
		}

		void ApplyQualityLevel()
		{
			switch (m_Quality)
			{
				case DataQuality.Custom:
					break;
				case DataQuality.VeryLow: // 18.62x smaller, 32.27 PSNR
					m_FormatPos = GaussianSplatAsset.VectorFormat.Norm11;
					m_FormatScale = GaussianSplatAsset.VectorFormat.Norm6;
					m_FormatColor = GaussianSplatAsset.ColorFormat.BC7;
					m_FormatSH = GaussianSplatAsset.SHFormat.Cluster4k;
					break;
				case DataQuality.Low: // 14.01x smaller, 35.17 PSNR
					m_FormatPos = GaussianSplatAsset.VectorFormat.Norm11;
					m_FormatScale = GaussianSplatAsset.VectorFormat.Norm6;
					m_FormatColor = GaussianSplatAsset.ColorFormat.Norm8x4;
					m_FormatSH = GaussianSplatAsset.SHFormat.Cluster16k;
					break;
				case DataQuality.Medium: // 5.14x smaller, 47.46 PSNR
					m_FormatPos = GaussianSplatAsset.VectorFormat.Norm11;
					m_FormatScale = GaussianSplatAsset.VectorFormat.Norm11;
					m_FormatColor = GaussianSplatAsset.ColorFormat.Norm8x4;
					m_FormatSH = GaussianSplatAsset.SHFormat.Norm6;
					break;
				case DataQuality.High: // 2.94x smaller, 57.77 PSNR
					m_FormatPos = GaussianSplatAsset.VectorFormat.Norm16;
					m_FormatScale = GaussianSplatAsset.VectorFormat.Norm16;
					m_FormatColor = GaussianSplatAsset.ColorFormat.Float16x4;
					m_FormatSH = GaussianSplatAsset.SHFormat.Norm11;
					break;
				case DataQuality.VeryHigh: // 1.05x smaller
					m_FormatPos = GaussianSplatAsset.VectorFormat.Float32;
					m_FormatScale = GaussianSplatAsset.VectorFormat.Float32;
					m_FormatColor = GaussianSplatAsset.ColorFormat.Float32x4;
					m_FormatSH = GaussianSplatAsset.SHFormat.Float32;
					break;
				default:
					throw new ArgumentOutOfRangeException();
			}
		}


		static T CreateOrReplaceAsset<T>(T asset, string path) where T : UnityEngine.Object
		{
			T result = AssetDatabase.LoadAssetAtPath<T>(path);
			if (result == null)
			{
				AssetDatabase.CreateAsset(asset, path);
				result = asset;
			}
			else
			{
				if (typeof(Mesh).IsAssignableFrom(typeof(T))) { (result as Mesh)?.Clear(); }
				EditorUtility.CopySerialized(asset, result);
			}
			return result;
		}

		async void CreateAsset()
		{
			m_ErrorMessage = null;
			if (string.IsNullOrWhiteSpace(m_InputFile))
			{
				m_ErrorMessage = $"Select input PLY/SPZ file";
				return;
			}

			if (string.IsNullOrWhiteSpace(m_OutputFolder) || !m_OutputFolder.StartsWith("Assets/"))
			{
				m_ErrorMessage = $"Output folder must be within project, was '{m_OutputFolder}'";
				return;
			}
			Directory.CreateDirectory(m_OutputFolder);

			EditorUtility.DisplayProgressBar(kProgressTitle, "Reading data files", 0.0f);
			GaussianSplatAsset.CameraInfo[] cameras = LoadJsonCamerasFile(m_InputFile, m_ImportCameras);
			using NativeArray<InputSplatData> inputSplats = LoadInputSplatFile(m_InputFile);
			if (inputSplats.Length == 0)
			{
				EditorUtility.ClearProgressBar();
				return;
			}

			// Use the runtime builder to create asset data
			var buildSettings = new GaussianSplatAssetBuilder.BuildSettings
			{
				formatPos = m_FormatPos,
				formatScale = m_FormatScale,
				formatColor = m_FormatColor,
				formatSH = m_FormatSH
			};

			bool ProgressCallback(string message, float progress)
			{
				EditorUtility.DisplayProgressBar(kProgressTitle, message, progress);
				return true;
			}

			var builder = new GaussianSplatAssetBuilder(buildSettings, ProgressCallback);
			var buildResult = await builder.BuildAsset(inputSplats);

			string baseName = Path.GetFileNameWithoutExtension(FilePickerControl.PathToDisplayString(m_InputFile));

			// Save the built data to files
			EditorUtility.DisplayProgressBar(kProgressTitle, "Saving data files", 0.85f);
			string pathChunk = $"{m_OutputFolder}/{baseName}_chk.bytes";
			string pathPos = $"{m_OutputFolder}/{baseName}_pos.bytes";
			string pathOther = $"{m_OutputFolder}/{baseName}_oth.bytes";
			string pathCol = $"{m_OutputFolder}/{baseName}_col.bytes";
			string pathSh = $"{m_OutputFolder}/{baseName}_shs.bytes";

			bool useChunks = buildResult.chunkData.IsCreated;
			if (useChunks)
				File.WriteAllBytes(pathChunk, buildResult.chunkData.ToArray());
			File.WriteAllBytes(pathPos, buildResult.posData.ToArray());
			File.WriteAllBytes(pathOther, buildResult.otherData.ToArray());
			File.WriteAllBytes(pathCol, buildResult.colorData.ToArray());
			File.WriteAllBytes(pathSh, buildResult.shData.ToArray());

			// Import the files so we can reference them
			EditorUtility.DisplayProgressBar(kProgressTitle, "Importing asset files", 0.90f);
			AssetDatabase.Refresh(ImportAssetOptions.ForceUncompressedImport);

			// Create the asset
			EditorUtility.DisplayProgressBar(kProgressTitle, "Creating asset", 0.95f);
			GaussianSplatAsset asset = ScriptableObject.CreateInstance<GaussianSplatAsset>();
			asset.Initialize(inputSplats.Length, m_FormatPos, m_FormatScale, m_FormatColor, m_FormatSH,
				buildResult.boundsMin, buildResult.boundsMax, cameras);
			asset.SetDataHash(buildResult.dataHash);
			asset.name = baseName;

			asset.SetAssetFiles(
				useChunks ? AssetDatabase.LoadAssetAtPath<TextAsset>(pathChunk) : null,
				AssetDatabase.LoadAssetAtPath<TextAsset>(pathPos),
				AssetDatabase.LoadAssetAtPath<TextAsset>(pathOther),
				AssetDatabase.LoadAssetAtPath<TextAsset>(pathCol),
				AssetDatabase.LoadAssetAtPath<TextAsset>(pathSh));

			var assetPath = $"{m_OutputFolder}/{baseName}.asset";
			var savedAsset = CreateOrReplaceAsset(asset, assetPath);

			EditorUtility.DisplayProgressBar(kProgressTitle, "Saving assets", 0.99f);
			AssetDatabase.SaveAssets();
			EditorUtility.ClearProgressBar();

			// Cleanup build result
			buildResult.Dispose();

			Selection.activeObject = savedAsset;
		}

		NativeArray<InputSplatData> LoadInputSplatFile(string filePath)
		{
			NativeArray<InputSplatData> data = default;
			if (!File.Exists(filePath))
			{
				m_ErrorMessage = $"Did not find {filePath} file";
				return data;
			}
			try
			{
				GaussianFileReader.ReadFile(filePath, out data);
			}
			catch (Exception ex)
			{
				m_ErrorMessage = ex.Message;
			}
			return data;
		}

		static GaussianSplatAsset.CameraInfo[] LoadJsonCamerasFile(string curPath, bool doImport)
		{
			if (!doImport)
				return null;

			string camerasPath;
			while (true)
			{
				var dir = Path.GetDirectoryName(curPath);
				if (!Directory.Exists(dir))
					return null;
				camerasPath = $"{dir}/{kCamerasJson}";
				if (File.Exists(camerasPath))
					break;
				curPath = dir;
			}

			if (!File.Exists(camerasPath))
				return null;

			string json = File.ReadAllText(camerasPath);
			var jsonCameras = JSONParser.FromJson<List<JsonCamera>>(json);
			if (jsonCameras == null || jsonCameras.Count == 0)
				return null;

			var result = new GaussianSplatAsset.CameraInfo[jsonCameras.Count];
			for (var camIndex = 0; camIndex < jsonCameras.Count; camIndex++)
			{
				var jsonCam = jsonCameras[camIndex];
				var pos = new Vector3(jsonCam.position[0], jsonCam.position[1], jsonCam.position[2]);
				// the matrix is a "view matrix", not "camera matrix" lol
				var axisx = new Vector3(jsonCam.rotation[0][0], jsonCam.rotation[1][0], jsonCam.rotation[2][0]);
				var axisy = new Vector3(jsonCam.rotation[0][1], jsonCam.rotation[1][1], jsonCam.rotation[2][1]);
				var axisz = new Vector3(jsonCam.rotation[0][2], jsonCam.rotation[1][2], jsonCam.rotation[2][2]);

				axisy *= -1;
				axisz *= -1;

				var cam = new GaussianSplatAsset.CameraInfo
				{
					pos = pos,
					axisX = axisx,
					axisY = axisy,
					axisZ = axisz,
					fov = 25 //@TODO
				};
				result[camIndex] = cam;
			}

			return result;
		}

		[Serializable]
		public class JsonCamera
		{
			public int id;
			public string img_name;
			public int width;
			public int height;
			public float[] position;
			public float[][] rotation;
			public float fx;
			public float fy;
		}
	}
}
