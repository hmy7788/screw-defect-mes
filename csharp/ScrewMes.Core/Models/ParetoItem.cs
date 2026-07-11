namespace ScrewMes.Core.Models;

/// <summary>품질분석 Pareto 한 항목 (불량 유형별 건수 + 누적 비율).</summary>
public record ParetoItem(string DefectType, int Count, double CumulativePercent);
