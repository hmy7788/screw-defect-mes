using System.IO;

namespace ScrewMes.App;

/// <summary>실행 위치에서 레포 루트를 찾아 모델/데이터/히트맵 경로를 제공.</summary>
public static class AppPaths
{
    public static string RepoRoot { get; } = FindRepoRoot();

    /// <summary>csharp/model — meta.json + weights/onnx/model.onnx (export_onnx.py 산출물).</summary>
    public static string ModelDir => Path.Combine(RepoRoot, "csharp", "model");

    /// <summary>히트맵 PNG 저장 폴더 (gitignore).</summary>
    public static string HeatmapDir => Path.Combine(RepoRoot, "csharp", "runtime", "heatmaps");

    /// <summary>라인에 흘려보낼 이미지 출처.</summary>
    public static string FeedSource => Path.Combine(RepoRoot, "data", "screw", "test");

    private static string FindRepoRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !Directory.Exists(Path.Combine(dir.FullName, "data", "screw", "test")))
            dir = dir.Parent;
        return dir?.FullName
            ?? throw new DirectoryNotFoundException("레포 루트(data/screw/test)를 찾지 못했습니다.");
    }
}
