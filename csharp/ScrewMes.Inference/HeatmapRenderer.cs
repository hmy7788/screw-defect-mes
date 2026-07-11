using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace ScrewMes.Inference;

/// <summary>원본 위에 anomaly_map(320×320)을 jet 컬러맵으로 오버레이해 PNG 저장.
/// Python render_heatmap 포팅: 이상도 min-max 정규화 → alpha=0.75·a 비례 합성.</summary>
internal static class HeatmapRenderer
{
    public static void Save(Image<Rgb24> source, float[] anomalyMap, int size, string outPath)
    {
        // min-max 정규화
        float min = float.MaxValue, max = float.MinValue;
        foreach (var v in anomalyMap) { if (v < min) min = v; if (v > max) max = v; }
        var range = max - min + 1e-9f;

        using var baseImg = source.Clone(c => c.Resize(size, size));
        baseImg.ProcessPixelRows(accessor =>
        {
            for (var y = 0; y < size; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (var x = 0; x < size; x++)
                {
                    var a = (anomalyMap[y * size + x] - min) / range;        // 0~1
                    var (hr, hg, hb) = JetColormap.Map(a);
                    var alpha = 0.75 * a;                                     // 이상도 비례 불투명도
                    ref var px = ref row[x];
                    px.R = (byte)((1 - alpha) * px.R + alpha * hr);
                    px.G = (byte)((1 - alpha) * px.G + alpha * hg);
                    px.B = (byte)((1 - alpha) * px.B + alpha * hb);
                }
            }
        });

        Directory.CreateDirectory(Path.GetDirectoryName(outPath)!);
        baseImg.SaveAsPng(outPath);
    }
}
