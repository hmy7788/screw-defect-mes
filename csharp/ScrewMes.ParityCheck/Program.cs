// C# ONNX 추론이 Python predict_one 과 일치하는지 검증 (M0의 C#측 go 확인).
// export_onnx.py 가 남긴 parity_samples.json 의 py_score/verdict 를 정답으로 비교한다.
using System.Text.Json;
using ScrewMes.Core;
using ScrewMes.Inference;

static string FindRepoRoot()
{
    var dir = new DirectoryInfo(AppContext.BaseDirectory);
    while (dir is not null && !Directory.Exists(Path.Combine(dir.FullName, "data", "screw", "test")))
        dir = dir.Parent;
    return dir?.FullName ?? throw new DirectoryNotFoundException("레포 루트(data/screw/test)를 못 찾음");
}

var root = FindRepoRoot();
var modelDir = Path.Combine(root, "csharp", "model");
var testDir = Path.Combine(root, "data", "screw", "test");
var samplesPath = Path.Combine(modelDir, "parity_samples.json");

if (!File.Exists(samplesPath))
{
    Console.WriteLine($"[오류] parity_samples.json 없음: {samplesPath}\n  -> 먼저 `python -m mes.core.export_onnx` 실행");
    return 1;
}

var samples = JsonSerializer.Deserialize<List<Sample>>(File.ReadAllText(samplesPath))!;
var heatmapDir = Path.Combine(root, "csharp", "runtime", "_paritycheck");
using var inf = new OnnxScrewInferencer(modelDir, heatmapDir);

double maxErr = 0;
int agree = 0, n = 0, rendered = 0;
Console.WriteLine($"{"image_id",-28} {"py",8} {"c#",8} {"|Δ|",8}  verdict");
foreach (var s in samples)
{
    var path = ResolveImage(testDir, s.image_id);
    if (path is null) { Console.WriteLine($"{s.image_id,-28} (이미지 못 찾음 — 건너뜀)"); continue; }

    var render = s.verdict == "bad";                  // 불량은 히트맵까지 생성해 렌더 경로도 검증
    var r = inf.Predict(path, render);
    if (render && r.HeatmapPath is not null && File.Exists(r.HeatmapPath)) rendered++;
    var err = Math.Abs(r.Score - s.py_score);
    maxErr = Math.Max(maxErr, err);
    var vmatch = r.Verdict.ToString().ToLower() == s.verdict;
    if (vmatch) agree++;
    n++;
    Console.WriteLine($"{s.image_id,-28} {s.py_score,8:F4} {r.Score,8:F4} {err,8:F5}  " +
                      $"{(vmatch ? "OK" : $"불일치({r.Verdict}/{s.verdict})")}");
}

Console.WriteLine($"\n표본 {n}장  최대|Δ|={maxErr:F5}  판정일치 {agree}/{n}  히트맵 {rendered}장 생성 ({heatmapDir})");
var ok = maxErr <= 0.01 && agree == n;
Console.WriteLine(ok ? "✅ C# ONNX 추론 == Python (go 확정)" : "⚠️ 불일치 — 전처리/모델 경로 재검토");
return ok ? 0 : 2;

// image_id = "{folder}_{stem}" (folder엔 '_' 포함 가능) → 실제 png 경로로 복원
static string? ResolveImage(string testDir, string imageId)
{
    foreach (var dir in Directory.GetDirectories(testDir))
    {
        var folder = Path.GetFileName(dir);
        if (!imageId.StartsWith(folder + "_")) continue;
        var stem = imageId[(folder.Length + 1)..];
        var p = Path.Combine(dir, stem + ".png");
        if (File.Exists(p)) return p;
    }
    return null;
}

record Sample(string image_id, double onnx_score, double py_score, string verdict);
