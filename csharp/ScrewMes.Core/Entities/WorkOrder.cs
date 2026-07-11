namespace ScrewMes.Core.Entities;

/// <summary>작업지시 — 생산의 최상위 단위. 하나의 WO가 여러 Lot을 가진다.</summary>
public class WorkOrder
{
    public int Id { get; set; }

    /// <summary>작업지시 번호 (예: WO-20260619-001).</summary>
    public string Code { get; set; } = "";

    /// <summary>생산 품목 (예: M3x8 나사).</summary>
    public string Product { get; set; } = "";

    /// <summary>계획 수량.</summary>
    public int PlannedQty { get; set; }

    public OrderStatus Status { get; set; } = OrderStatus.Open;

    public DateTime CreatedAt { get; set; } = DateTime.Now;

    public List<Lot> Lots { get; set; } = new();
}
