namespace ScrewMes.Inference;

/// <summary>matplotlib 'jet' 컬러맵 근사 (Python render_heatmap의 cm.jet 대응).
/// 입력 t∈[0,1] → (R,G,B) 0~255.</summary>
internal static class JetColormap
{
    public static (byte R, byte G, byte B) Map(double t)
    {
        t = Math.Clamp(t, 0.0, 1.0);
        var r = Clamp01(Math.Min(4 * t - 1.5, -4 * t + 4.5));
        var g = Clamp01(Math.Min(4 * t - 0.5, -4 * t + 3.5));
        var b = Clamp01(Math.Min(4 * t + 0.5, -4 * t + 2.5));
        return ((byte)(r * 255), (byte)(g * 255), (byte)(b * 255));
    }

    private static double Clamp01(double v) => Math.Clamp(v, 0.0, 1.0);
}
