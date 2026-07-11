namespace ScrewMes.App;

/// <summary>소스 폴더명 → 한글 유형 라벨 (히트맵/포폴 표기와 일치).</summary>
public static class DefectLabels
{
    private static readonly Dictionary<string, string> Map = new()
    {
        ["manipulated_front"] = "Type1 · 나사끝",
        ["scratch_head"] = "Type2 · 나사머리",
        ["scratch_neck"] = "Type3 · 나사목",
        ["thread_side"] = "Type4 · 나사산 측면",
        ["thread_top"] = "Type5 · 나사산 상단",
        ["good"] = "정상",
    };

    public static string Of(string? folder)
        => folder is null ? "정상(오탐)"
         : Map.TryGetValue(folder, out var v) ? v : folder;
}
