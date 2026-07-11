namespace ScrewMes.Core.Entities;

/// <summary>Lot — 한 작업지시 안에서 흘려보낸 검사 묶음(추적성의 중간 노드).</summary>
public class Lot
{
    public int Id { get; set; }

    /// <summary>Lot 번호 (예: LOT-20260619-1530).</summary>
    public string LotNo { get; set; } = "";

    public int WorkOrderId { get; set; }
    public WorkOrder? WorkOrder { get; set; }

    public DateTime StartedAt { get; set; } = DateTime.Now;

    public LotStatus Status { get; set; } = LotStatus.Open;

    public List<Inspection> Inspections { get; set; } = new();
}
