using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Design;

namespace ScrewMes.Data;

/// <summary>`dotnet ef migrations` 가 디자인타임에 DbContext를 만들 때 사용.</summary>
public class MesDbContextFactory : IDesignTimeDbContextFactory<MesDbContext>
{
    public MesDbContext CreateDbContext(string[] args)
    {
        var options = new DbContextOptionsBuilder<MesDbContext>()
            .UseSqlServer(DbConfig.ConnectionString)
            .Options;
        return new MesDbContext(options);
    }
}
