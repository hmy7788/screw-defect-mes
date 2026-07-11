namespace ScrewMes.Core.Models;

/// <summary>생산현황 집계 KPI.</summary>
public record ProductionStats(int Total, int Good, int Bad)
{
    public double DefectRate => Total == 0 ? 0.0 : (double)Bad / Total;
}
