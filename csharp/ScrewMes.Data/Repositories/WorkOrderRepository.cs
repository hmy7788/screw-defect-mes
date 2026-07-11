using Microsoft.EntityFrameworkCore;
using ScrewMes.Core;
using ScrewMes.Core.Abstractions;
using ScrewMes.Core.Entities;

namespace ScrewMes.Data.Repositories;

public class WorkOrderRepository : IWorkOrderRepository
{
    private readonly IDbContextFactory<MesDbContext> _factory;
    public WorkOrderRepository(IDbContextFactory<MesDbContext> factory) => _factory = factory;

    public async Task<WorkOrder> CreateAsync(string product, int plannedQty)
    {
        await using var db = await _factory.CreateDbContextAsync();
        var today = DateTime.Now.ToString("yyyyMMdd");
        var seq = await db.WorkOrders.CountAsync(w => w.Code.StartsWith("WO-" + today)) + 1;
        var wo = new WorkOrder
        {
            Code = $"WO-{today}-{seq:D3}",
            Product = product,
            PlannedQty = plannedQty,
            Status = OrderStatus.Open,
            CreatedAt = DateTime.Now,
        };
        db.WorkOrders.Add(wo);
        await db.SaveChangesAsync();
        return wo;
    }

    public async Task<Lot> StartLotAsync(int workOrderId)
    {
        await using var db = await _factory.CreateDbContextAsync();
        var wo = await db.WorkOrders.FindAsync(workOrderId)
                 ?? throw new InvalidOperationException($"작업지시 {workOrderId} 없음");
        wo.Status = OrderStatus.Running;
        var lot = new Lot
        {
            WorkOrderId = workOrderId,
            LotNo = $"LOT-{DateTime.Now:yyyyMMdd-HHmmss}",
            StartedAt = DateTime.Now,
            Status = LotStatus.Open,
        };
        db.Lots.Add(lot);
        await db.SaveChangesAsync();
        return lot;
    }

    public async Task<IReadOnlyList<WorkOrder>> ListAsync()
    {
        await using var db = await _factory.CreateDbContextAsync();
        return await db.WorkOrders.AsNoTracking().OrderByDescending(w => w.Id).ToListAsync();
    }

    public async Task<IReadOnlyList<Lot>> LotsAsync(int workOrderId)
    {
        await using var db = await _factory.CreateDbContextAsync();
        return await db.Lots.AsNoTracking()
            .Where(l => l.WorkOrderId == workOrderId)
            .OrderByDescending(l => l.Id).ToListAsync();
    }

    public async Task<IReadOnlyList<Inspection>> LotInspectionsAsync(int lotId)
    {
        await using var db = await _factory.CreateDbContextAsync();
        return await db.Inspections.AsNoTracking()
            .Where(i => i.LotId == lotId)
            .OrderBy(i => i.Id).ToListAsync();
    }
}
