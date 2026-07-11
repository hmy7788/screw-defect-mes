using Microsoft.EntityFrameworkCore;
using ScrewMes.Core;
using ScrewMes.Core.Abstractions;
using ScrewMes.Core.Entities;
using ScrewMes.Core.Models;

namespace ScrewMes.Data.Repositories;

public class InspectionRepository : IInspectionRepository
{
    private readonly IDbContextFactory<MesDbContext> _factory;
    public InspectionRepository(IDbContextFactory<MesDbContext> factory) => _factory = factory;

    public async Task<int> InsertAsync(Inspection inspection)
    {
        await using var db = await _factory.CreateDbContextAsync();
        db.Inspections.Add(inspection);
        await db.SaveChangesAsync();
        return inspection.Id;
    }

    public async Task<IReadOnlyList<Inspection>> RecentAsync(int limit = 20, bool onlyBad = false)
    {
        await using var db = await _factory.CreateDbContextAsync();
        var q = db.Inspections.AsNoTracking();
        if (onlyBad) q = q.Where(i => i.Verdict == Verdict.Bad);
        return await q.OrderByDescending(i => i.Id).Take(limit).ToListAsync();
    }

    public async Task<IReadOnlyList<Inspection>> HistoryAsync(int? limit = null)
    {
        await using var db = await _factory.CreateDbContextAsync();
        if (limit is null)
            return await db.Inspections.AsNoTracking().OrderBy(i => i.Id).ToListAsync();

        var last = await db.Inspections.AsNoTracking()
            .OrderByDescending(i => i.Id).Take(limit.Value).ToListAsync();
        last.Reverse();                                   // 최신 N건을 오래된→최신 순으로
        return last;
    }

    public async Task<ProductionStats> StatsAsync()
    {
        await using var db = await _factory.CreateDbContextAsync();
        var total = await db.Inspections.CountAsync();
        var bad = await db.Inspections.CountAsync(i => i.Verdict == Verdict.Bad);
        return new ProductionStats(total, total - bad, bad);
    }

    public async Task<IReadOnlyList<ParetoItem>> ParetoAsync()
    {
        await using var db = await _factory.CreateDbContextAsync();
        var groups = await db.Inspections.AsNoTracking()
            .Where(i => i.Verdict == Verdict.Bad)
            .GroupBy(i => i.DefectType)
            .Select(g => new { Type = g.Key, Count = g.Count() })
            .ToListAsync();

        var ordered = groups.OrderByDescending(g => g.Count).ToList();
        var total = ordered.Sum(g => g.Count);
        var items = new List<ParetoItem>();
        var cum = 0;
        foreach (var g in ordered)
        {
            cum += g.Count;
            var label = g.Type ?? "정상(오탐)";          // good 소스를 불량 판정 = 오탐
            items.Add(new ParetoItem(label, g.Count, total == 0 ? 0 : 100.0 * cum / total));
        }
        return items;
    }

    public async Task ResetAsync()
    {
        await using var db = await _factory.CreateDbContextAsync();
        // 데모 재시작: 자식(검사)→Lot→작업지시 순으로 비운다.
        await db.Database.ExecuteSqlRawAsync("DELETE FROM Inspections");
        await db.Database.ExecuteSqlRawAsync("DELETE FROM Lots");
        await db.Database.ExecuteSqlRawAsync("DELETE FROM WorkOrders");
    }
}
