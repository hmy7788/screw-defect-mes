using ScrewMes.Core.Entities;
using ScrewMes.Core.Models;

namespace ScrewMes.Core.Abstractions;

/// <summary>검사 기록 영속화 + 집계. Python repository.py 시그니처 계승 + Pareto 추가.</summary>
public interface IInspectionRepository
{
    Task<int> InsertAsync(Inspection inspection);

    /// <summary>최신순 N건 (onlyBad=true면 불량만).</summary>
    Task<IReadOnlyList<Inspection>> RecentAsync(int limit = 20, bool onlyBad = false);

    /// <summary>오래된→최신 순 전체(추세 차트용). limit이면 최근 N건만.</summary>
    Task<IReadOnlyList<Inspection>> HistoryAsync(int? limit = null);

    Task<ProductionStats> StatsAsync();

    /// <summary>불량(verdict=Bad)을 DefectType별로 집계한 Pareto. good 소스 오탐은 '정상(오탐)'.</summary>
    Task<IReadOnlyList<ParetoItem>> ParetoAsync();

    /// <summary>데모 재시작: 전체 삭제.</summary>
    Task ResetAsync();
}
