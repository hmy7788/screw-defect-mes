using System.Text.Json;

namespace ScrewMes.Inference;

/// <summary>export_onnx.py 가 남긴 meta.json. C#이 전처리·입출력명·임계값을 그대로 따른다.
///
/// 핵심(진단으로 확정): onnx 그래프가 Resize+Normalize+점수정규화를 모두 내장 →
/// 입력은 '원본 RGB를 255로 나눈 [0,1] CHW 텐서'만 넣으면 되고, pred_score 출력에
/// 임계값(0.513)을 바로 적용하면 Python predict_one 과 동일(오차 ~5e-5).</summary>
public sealed class InferenceMeta
{
    public int ImageSize { get; init; } = 320;
    public double Threshold { get; init; } = 0.513;
    public string Input { get; init; } = "input";
    public string ScoreOutput { get; init; } = "pred_score";
    public string MapOutput { get; init; } = "anomaly_map";

    public static InferenceMeta Load(string path)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        var root = doc.RootElement;
        var onnx = root.GetProperty("onnx");
        return new InferenceMeta
        {
            ImageSize = root.GetProperty("image_size").GetInt32(),
            Threshold = root.GetProperty("threshold").GetDouble(),
            Input = onnx.GetProperty("input").GetString()!,
            ScoreOutput = onnx.GetProperty("score_output").GetString()!,
            MapOutput = onnx.GetProperty("map_output").GetString()!,
        };
    }
}
