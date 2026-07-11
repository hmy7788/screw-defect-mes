using ScrewMes.Core.Entities;

namespace ScrewMes.Core.Abstractions;

/// <summary>작업지시/Lot 관리 + 추적성(genealogy) 조회.</summary>
public interface IWorkOrderRepository
{
    Task<WorkOrder> CreateAsync(string product, int plannedQty);

    /// <summary>작업지시에 새 Lot을 열고 WO를 Running으로.</summary>
    Task<Lot> StartLotAsync(int workOrderId);

    Task<IReadOnlyList<WorkOrder>> ListAsync();

    Task<IReadOnlyList<Lot>> LotsAsync(int workOrderId);

    /// <summary>Lot 하위 검사 이력(추적성 드릴다운).</summary>
    Task<IReadOnlyList<Inspection>> LotInspectionsAsync(int lotId);
}
