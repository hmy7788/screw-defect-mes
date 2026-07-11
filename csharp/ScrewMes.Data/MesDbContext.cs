using Microsoft.EntityFrameworkCore;
using ScrewMes.Core;
using ScrewMes.Core.Entities;

namespace ScrewMes.Data;

public class MesDbContext : DbContext
{
    public MesDbContext(DbContextOptions<MesDbContext> options) : base(options) { }

    public DbSet<WorkOrder> WorkOrders => Set<WorkOrder>();
    public DbSet<Lot> Lots => Set<Lot>();
    public DbSet<Inspection> Inspections => Set<Inspection>();

    protected override void OnModelCreating(ModelBuilder b)
    {
        b.Entity<WorkOrder>(e =>
        {
            e.Property(x => x.Code).HasMaxLength(40);
            e.HasIndex(x => x.Code).IsUnique();
            e.Property(x => x.Product).HasMaxLength(80);
            e.Property(x => x.Status).HasConversion<string>().HasMaxLength(16);
        });

        b.Entity<Lot>(e =>
        {
            e.Property(x => x.LotNo).HasMaxLength(40);
            e.Property(x => x.Status).HasConversion<string>().HasMaxLength(16);
            e.HasOne(x => x.WorkOrder).WithMany(w => w.Lots)
                .HasForeignKey(x => x.WorkOrderId).OnDelete(DeleteBehavior.Cascade);
        });

        b.Entity<Inspection>(e =>
        {
            e.Property(x => x.Verdict).HasConversion<string>().HasMaxLength(8);
            e.Property(x => x.ImageId).HasMaxLength(80);
            e.Property(x => x.DefectType).HasMaxLength(40);
            e.HasIndex(x => x.Ts);
            e.HasOne(x => x.Lot).WithMany(l => l.Inspections)
                .HasForeignKey(x => x.LotId).OnDelete(DeleteBehavior.Cascade);
        });
    }
}
