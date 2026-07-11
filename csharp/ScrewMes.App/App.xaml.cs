using System.Windows;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.DependencyInjection;
using ScrewMes.Core.Abstractions;
using ScrewMes.Data;
using ScrewMes.Data.Repositories;
using ScrewMes.Inference;
using ScrewMes.Simulator;
using ScrewMes.App.ViewModels;

namespace ScrewMes.App;

public partial class App : Application
{
    protected override void OnStartup(StartupEventArgs e)
    {
        base.OnStartup(e);
        try
        {
            var services = new ServiceCollection();
            services.AddDbContextFactory<MesDbContext>(o => o.UseSqlServer(DbConfig.ConnectionString));
            services.AddSingleton<IInspectionRepository, InspectionRepository>();
            services.AddSingleton<IWorkOrderRepository, WorkOrderRepository>();
            services.AddSingleton<IScrewInferencer>(_ =>
                new OnnxScrewInferencer(AppPaths.ModelDir, AppPaths.HeatmapDir));
            services.AddSingleton(sp => new LineFeeder(
                sp.GetRequiredService<IScrewInferencer>(),
                sp.GetRequiredService<IInspectionRepository>(),
                AppPaths.FeedSource));
            services.AddSingleton<MainViewModel>();

            var provider = services.BuildServiceProvider();

            // DB 생성/마이그레이션 (LocalDB)
            using (var db = provider.GetRequiredService<IDbContextFactory<MesDbContext>>().CreateDbContext())
                db.Database.Migrate();

            var vm = provider.GetRequiredService<MainViewModel>();
            var window = new MainWindow { DataContext = vm };
            window.Show();
            _ = vm.InitializeAsync();
        }
        catch (Exception ex)
        {
            MessageBox.Show(
                $"시작 실패:\n{ex.Message}\n\n" +
                "확인:\n • python -m mes.core.export_onnx (model.onnx/meta.json 생성)\n" +
                " • SQL Server LocalDB 설치 여부",
                "ScrewMes", MessageBoxButton.OK, MessageBoxImage.Error);
            Shutdown();
        }
    }
}
