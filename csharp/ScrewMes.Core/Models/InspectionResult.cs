namespace ScrewMes.Core.Models;

/// <summary>추론 1건의 결과(저장 전). Python predict_one 반환 dict와 동형.</summary>
public record InspectionResult(
    string ImageId,
    DateTime Ts,
    Verdict Verdict,
    double Score,
    string? HeatmapPath);
