using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using ScrewMes.Core;
using ScrewMes.Core.Abstractions;
using ScrewMes.Core.Models;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace ScrewMes.Inference;

/// <summary>ONNX Runtime 네이티브 추론 (외부 Python 프로세스 없음).
///
/// model.onnx 는 anomalib export 산출물로 전처리·후처리가 내장돼 있어,
/// 입력에 '원본 RGB/255 [0,1] CHW' 텐서만 넣으면 pred_score(정규화 이상점수)·anomaly_map 이 나온다.
/// pred_score ≥ threshold(0.513) → Bad.</summary>
public sealed class OnnxScrewInferencer : IScrewInferencer, IDisposable
{
    private readonly InferenceSession _session;
    private readonly InferenceMeta _meta;
    private readonly string _heatmapDir;

    /// <param name="modelDir">csharp/model (meta.json + weights/onnx/model.onnx 위치).</param>
    /// <param name="heatmapDir">히트맵 PNG 저장 폴더.</param>
    public OnnxScrewInferencer(string modelDir, string heatmapDir)
    {
        var metaPath = Path.Combine(modelDir, "meta.json");
        if (!File.Exists(metaPath))
            throw new FileNotFoundException(
                $"meta.json 이 없습니다: {metaPath}\n  -> 먼저 `python -m mes.core.export_onnx` 실행.");
        _meta = InferenceMeta.Load(metaPath);

        var onnxPath = Directory.EnumerateFiles(modelDir, "*.onnx", SearchOption.AllDirectories)
            .OrderBy(p => p.Length).FirstOrDefault()
            ?? throw new FileNotFoundException(
                $"model.onnx 가 없습니다 ({modelDir}).\n  -> 먼저 `python -m mes.core.export_onnx` 실행.");

        _session = new InferenceSession(onnxPath);
        _heatmapDir = heatmapDir;
        Directory.CreateDirectory(_heatmapDir);
    }

    public InspectionResult Predict(string imagePath, bool render = true)
    {
        var imageId = $"{Directory.GetParent(imagePath)!.Name}_{Path.GetFileNameWithoutExtension(imagePath)}";

        using var image = Image.Load<Rgb24>(imagePath);
        var input = ToTensor(image);                              // 원본 해상도 그대로 (그래프가 리사이즈)

        var wanted = render ? new[] { _meta.ScoreOutput, _meta.MapOutput }
                            : new[] { _meta.ScoreOutput };
        var inputs = new[] { NamedOnnxValue.CreateFromTensor(_meta.Input, input) };
        using var results = _session.Run(inputs, wanted);

        var score = results.First(r => r.Name == _meta.ScoreOutput).AsTensor<float>().GetValue(0);
        var verdict = score >= _meta.Threshold ? Verdict.Bad : Verdict.Good;

        string? heatmapPath = null;
        if (render)
        {
            var map = results.First(r => r.Name == _meta.MapOutput).AsTensor<float>();
            heatmapPath = Path.Combine(_heatmapDir, $"{imageId}.png");
            HeatmapRenderer.Save(image, map.ToArray(), _meta.ImageSize, heatmapPath);
        }

        return new InspectionResult(imageId, DateTime.Now, verdict, Math.Round(score, 4), heatmapPath);
    }

    /// <summary>원본 RGB → DenseTensor[1,3,H,W], 값 = 픽셀/255 (정규화·리사이즈는 onnx 그래프 담당).</summary>
    private static DenseTensor<float> ToTensor(Image<Rgb24> image)
    {
        int h = image.Height, w = image.Width, plane = h * w;
        var data = new float[3 * plane];
        image.ProcessPixelRows(accessor =>
        {
            for (var y = 0; y < h; y++)
            {
                var row = accessor.GetRowSpan(y);
                var off = y * w;
                for (var x = 0; x < w; x++)
                {
                    var px = row[x];
                    data[off + x] = px.R / 255f;                  // R 평면
                    data[plane + off + x] = px.G / 255f;          // G 평면
                    data[2 * plane + off + x] = px.B / 255f;      // B 평면
                }
            }
        });
        return new DenseTensor<float>(data, new[] { 1, 3, h, w });
    }

    public void Dispose() => _session.Dispose();
}
