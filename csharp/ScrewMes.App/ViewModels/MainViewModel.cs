using System.Collections.ObjectModel;
using System.Windows;
using System.Windows.Threading;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using LiveChartsCore;
using LiveChartsCore.Measure;
using LiveChartsCore.SkiaSharpView;
using LiveChartsCore.SkiaSharpView.Painting;
using SkiaSharp;
using ScrewMes.Core;
using ScrewMes.Core.Abstractions;
using ScrewMes.Core.Entities;
using ScrewMes.Simulator;

namespace ScrewMes.App.ViewModels;

public record GalleryItem(string? HeatmapPath, string Caption);
public record ParetoRow(string Label, int Count, double CumulativePercent);
public record TraceRow(string Time, string ImageId, string Verdict, double Score, string Type, string? HeatmapPath);

/// <summary>관제 화면 전체를 구동하는 ViewModel — DB 폴링(1.5s)으로 KPI/차트/갤러리/Pareto를
/// 갱신하고, 라인 시뮬레이터를 백그라운드로 가동/정지한다.
///
/// 차트는 시리즈 객체를 '한 번만' 만들고 값(ObservableCollection)을 제자리 갱신한다.
/// 매 틱 시리즈를 새로 만들면 그래프가 처음부터 다시 그려져 '튀기' 때문. 또 새 검사가 없으면
/// 새로고침 자체를 건너뛴다(정지 상태에서 그래프가 가만히 있도록).</summary>
public partial class MainViewModel : ObservableObject
{
    private readonly IInspectionRepository _inspections;
    private readonly IWorkOrderRepository _workOrders;
    private readonly LineFeeder _feeder;
    private readonly DispatcherTimer _timer;
    private CancellationTokenSource? _cts;
    private bool _refreshing;
    private int _lastMaxId = -1;       // 새 검사 없으면 갱신 생략(정지 시 그래프 고정)
    private int _lastBadCount = -1;    // 불량 수 바뀔 때만 히트맵 갤러리 재구성

    // 차트 값(제자리 갱신) — 시리즈는 생성자에서 1회만 만든다
    private readonly ObservableCollection<double> _scoreVals = new();
    private readonly ObservableCollection<double> _thrVals = new();
    private readonly ObservableCollection<double> _cumVals = new();
    private readonly ObservableCollection<double> _rollVals = new();
    private readonly ObservableCollection<int> _paretoCounts = new();
    private readonly ObservableCollection<double> _paretoCum = new();
    private readonly Axis _paretoXAxis = new() { LabelsRotation = 15 };

    public MainViewModel(IInspectionRepository inspections, IWorkOrderRepository workOrders, LineFeeder feeder)
    {
        _inspections = inspections;
        _workOrders = workOrders;
        _feeder = feeder;

        // 시리즈 1회 생성 (이후 값만 제자리 갱신)
        ScoreSeries = new ISeries[]
        {
            new LineSeries<double> { Values = _scoreVals, Name = "이상점수", GeometrySize = 0,
                Stroke = new SolidColorPaint(SKColors.SteelBlue, 2), Fill = null },
            new LineSeries<double> { Values = _thrVals, Name = "임계값 0.513", GeometrySize = 0,
                Stroke = new SolidColorPaint(SKColors.IndianRed, 1), Fill = null },
        };
        DefectRateSeries = new ISeries[]
        {
            new LineSeries<double> { Values = _cumVals, Name = "누적 불량률%", GeometrySize = 0,
                Stroke = new SolidColorPaint(SKColors.DimGray, 2), Fill = null },
            new LineSeries<double> { Values = _rollVals, Name = "이동 불량률%(20)", GeometrySize = 0,
                Stroke = new SolidColorPaint(SKColors.DarkOrange, 2), Fill = null },
        };
        ParetoSeries = new ISeries[]
        {
            new ColumnSeries<int> { Values = _paretoCounts, Name = "건수",
                Fill = new SolidColorPaint(SKColors.SteelBlue) },
            new LineSeries<double> { Values = _paretoCum, Name = "누적%", ScalesYAt = 1,
                GeometrySize = 8, Fill = null, Stroke = new SolidColorPaint(SKColors.DarkOrange, 2) },
        };
        ParetoXAxes = new[] { _paretoXAxis };
        ParetoYAxes = new[]
        {
            new Axis { Name = "건수", MinLimit = 0 },
            new Axis { Name = "누적%", MinLimit = 0, MaxLimit = 100, Position = AxisPosition.End },
        };

        _timer = new DispatcherTimer { Interval = TimeSpan.FromSeconds(1.5) };
        _timer.Tick += async (_, _) => await RefreshAsync();
    }

    // ----- 입력/설정 -----
    [ObservableProperty] private string _newProduct = "M3x8 나사";
    [ObservableProperty] private int _newQty = 1000;
    [ObservableProperty] private double _goodRatio = 0.85;
    [ObservableProperty] private double _feedInterval = 1.0;

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(StartLineCommand))]
    [NotifyCanExecuteChangedFor(nameof(StopLineCommand))]
    private bool _lineRunning;

    [ObservableProperty] private string _statusText = "대기";

    // ----- 대시보드 KPI -----
    [ObservableProperty] private int _total;
    [ObservableProperty] private int _goodCount;
    [ObservableProperty] private int _badCount;
    [ObservableProperty] private string _defectRateText = "0.0%";
    [ObservableProperty] private string _latestAlarmText = "검사 대기 중";
    [ObservableProperty] private bool _latestIsBad;

    // ----- 차트 (시리즈는 불변, 값만 갱신) -----
    [ObservableProperty] private ISeries[] _scoreSeries = Array.Empty<ISeries>();
    [ObservableProperty] private ISeries[] _defectRateSeries = Array.Empty<ISeries>();
    [ObservableProperty] private ISeries[] _paretoSeries = Array.Empty<ISeries>();
    [ObservableProperty] private Axis[] _paretoXAxes = Array.Empty<Axis>();
    [ObservableProperty] private Axis[] _paretoYAxes = Array.Empty<Axis>();

    // ----- 컬렉션 -----
    public ObservableCollection<GalleryItem> RecentBad { get; } = new();
    public ObservableCollection<WorkOrder> WorkOrders { get; } = new();
    public ObservableCollection<Lot> Lots { get; } = new();
    public ObservableCollection<TraceRow> LotInspections { get; } = new();
    public ObservableCollection<ParetoRow> ParetoRows { get; } = new();

    [ObservableProperty] private WorkOrder? _selectedWorkOrder;
    [ObservableProperty] private Lot? _selectedLot;
    [ObservableProperty] private TraceRow? _selectedTrace;

    partial void OnSelectedWorkOrderChanged(WorkOrder? value)
    {
        if (value is not null) _ = LoadLotsAsync(value.Id);
    }

    partial void OnSelectedLotChanged(Lot? value)
    {
        if (value is not null) _ = LoadLotInspectionsAsync(value.Id);
    }

    // ----- 수명주기 -----
    public async Task InitializeAsync()
    {
        await LoadWorkOrdersAsync();
        await RefreshAsync();
        _timer.Start();
    }

    private async Task RefreshAsync()
    {
        if (_refreshing) return;
        _refreshing = true;
        try
        {
            var latest = (await _inspections.RecentAsync(1)).FirstOrDefault();
            var maxId = latest?.Id ?? 0;
            if (maxId == _lastMaxId) return;        // 새 검사 없음 → UI 손대지 않음(튀지 않음)
            _lastMaxId = maxId;

            var stats = await _inspections.StatsAsync();
            Total = stats.Total;
            GoodCount = stats.Good;
            BadCount = stats.Bad;
            DefectRateText = stats.DefectRate.ToString("P1");

            if (latest is not null)
            {
                LatestIsBad = latest.Verdict == Verdict.Bad;
                LatestAlarmText = $"{(LatestIsBad ? "⛔ 불량" : "✅ 양품")}   {latest.ImageId}   " +
                                  $"점수 {latest.Score:F3}   ({latest.Ts:HH:mm:ss})";
            }

            UpdateTrendCharts(await _inspections.HistoryAsync(80));

            if (BadCount != _lastBadCount)          // 불량 수 변할 때만 갤러리·Pareto 재구성
            {
                _lastBadCount = BadCount;
                RecentBad.Clear();
                foreach (var i in await _inspections.RecentAsync(8, onlyBad: true))
                    RecentBad.Add(new GalleryItem(i.HeatmapPath, $"{DefectLabels.Of(i.DefectType)}  ({i.Score:F2})"));
                UpdatePareto(await _inspections.ParetoAsync());
            }

            if (SelectedLot is not null) await LoadLotInspectionsAsync(SelectedLot.Id, force: false);
        }
        finally { _refreshing = false; }
    }

    // ----- 차트 값 제자리 갱신 -----
    private void UpdateTrendCharts(IReadOnlyList<Inspection> history)
    {
        var n = history.Count;
        var scores = new double[n];
        var thr = new double[n];
        var cum = new double[n];
        var roll = new double[n];
        const int win = 20;
        var bad = 0;
        for (var k = 0; k < n; k++)
        {
            scores[k] = history[k].Score;
            thr[k] = 0.513;
            if (history[k].Verdict == Verdict.Bad) bad++;
            cum[k] = 100.0 * bad / (k + 1);
            var start = Math.Max(0, k - win + 1);
            int cnt = 0, bb = 0;
            for (var j = start; j <= k; j++) { cnt++; if (history[j].Verdict == Verdict.Bad) bb++; }
            roll[k] = 100.0 * bb / cnt;
        }
        Sync(_scoreVals, scores);
        Sync(_thrVals, thr);
        Sync(_cumVals, cum);
        Sync(_rollVals, roll);
    }

    private void UpdatePareto(IReadOnlyList<Core.Models.ParetoItem> pareto)
    {
        ParetoRows.Clear();
        var counts = new int[pareto.Count];
        var cum = new double[pareto.Count];
        var labels = new List<string>(pareto.Count);
        for (var i = 0; i < pareto.Count; i++)
        {
            var label = DefectLabels.Of(pareto[i].DefectType);
            counts[i] = pareto[i].Count;
            cum[i] = pareto[i].CumulativePercent;
            labels.Add(label);
            ParetoRows.Add(new ParetoRow(label, pareto[i].Count, pareto[i].CumulativePercent));
        }
        Sync(_paretoCounts, counts);
        Sync(_paretoCum, cum);
        _paretoXAxis.Labels = labels;          // 축 객체는 유지하고 라벨만 교체
    }

    /// <summary>ObservableCollection을 target과 동기화(길이/값 제자리 변경) → 매끄러운 차트 갱신.</summary>
    private static void Sync(ObservableCollection<double> dst, IReadOnlyList<double> src)
    {
        for (var i = 0; i < src.Count; i++)
        {
            if (i < dst.Count) { if (dst[i] != src[i]) dst[i] = src[i]; }
            else dst.Add(src[i]);
        }
        while (dst.Count > src.Count) dst.RemoveAt(dst.Count - 1);
    }

    private static void Sync(ObservableCollection<int> dst, IReadOnlyList<int> src)
    {
        for (var i = 0; i < src.Count; i++)
        {
            if (i < dst.Count) { if (dst[i] != src[i]) dst[i] = src[i]; }
            else dst.Add(src[i]);
        }
        while (dst.Count > src.Count) dst.RemoveAt(dst.Count - 1);
    }

    // ----- 데이터 로드 -----
    private async Task LoadWorkOrdersAsync()
    {
        WorkOrders.Clear();
        foreach (var w in await _workOrders.ListAsync()) WorkOrders.Add(w);
    }

    private async Task LoadLotsAsync(int workOrderId)
    {
        Lots.Clear();
        foreach (var l in await _workOrders.LotsAsync(workOrderId)) Lots.Add(l);
    }

    /// <param name="force">true=전체 재구성(Lot 선택 변경 시), false=새 행만 append(주기 갱신 시).</param>
    private async Task LoadLotInspectionsAsync(int lotId, bool force = true)
    {
        var rows = await _workOrders.LotInspectionsAsync(lotId);
        if (!force && rows.Count == LotInspections.Count) return;          // 변화 없음
        if (!force && rows.Count > LotInspections.Count)                   // 같은 Lot에 행 추가됨 → 꼬리만 append
        {
            for (var i = LotInspections.Count; i < rows.Count; i++)
                LotInspections.Add(ToRow(rows[i]));
            return;
        }
        LotInspections.Clear();                                           // 다른 Lot 선택 등 → 전체 교체
        foreach (var i in rows) LotInspections.Add(ToRow(i));
    }

    private static TraceRow ToRow(Inspection i) => new(
        i.Ts.ToString("HH:mm:ss"), i.ImageId,
        i.Verdict == Verdict.Bad ? "불량" : "양품",
        i.Score, DefectLabels.Of(i.DefectType), i.HeatmapPath);

    // ----- 커맨드 -----
    [RelayCommand]
    private async Task CreateWorkOrder()
    {
        var wo = await _workOrders.CreateAsync(NewProduct, NewQty);
        await LoadWorkOrdersAsync();
        SelectedWorkOrder = WorkOrders.FirstOrDefault(w => w.Id == wo.Id);
    }

    [RelayCommand(CanExecute = nameof(CanStart))]
    private async Task StartLine()
    {
        var wo = SelectedWorkOrder;
        if (wo is null)
        {
            wo = await _workOrders.CreateAsync(NewProduct, NewQty);
            await LoadWorkOrdersAsync();
            SelectedWorkOrder = WorkOrders.FirstOrDefault(w => w.Id == wo.Id);
            wo = SelectedWorkOrder!;
        }

        var lot = await _workOrders.StartLotAsync(wo.Id);
        await LoadLotsAsync(wo.Id);
        SelectedLot = Lots.FirstOrDefault(l => l.Id == lot.Id);

        _cts = new CancellationTokenSource();
        LineRunning = true;
        StatusText = $"가동 중 — {lot.LotNo}";

        var token = _cts.Token;
        var (lotId, iv, gr) = (lot.Id, FeedInterval, GoodRatio);
        _ = Task.Run(async () =>
        {
            try { await _feeder.RunAsync(lotId, null, iv, gr, null, token); }
            catch (Exception ex)
            {
                Application.Current.Dispatcher.Invoke(() => StatusText = "오류: " + ex.Message);
            }
        });
    }

    private bool CanStart() => !LineRunning;

    [RelayCommand(CanExecute = nameof(CanStop))]
    private void StopLine()
    {
        _cts?.Cancel();
        LineRunning = false;
        StatusText = "정지";
    }

    private bool CanStop() => LineRunning;

    [RelayCommand]
    private async Task ResetDb()
    {
        StopLine();
        await _inspections.ResetAsync();
        WorkOrders.Clear(); Lots.Clear(); LotInspections.Clear();
        RecentBad.Clear(); ParetoRows.Clear();
        SelectedWorkOrder = null; SelectedLot = null;
        _lastMaxId = -1; _lastBadCount = -1;        // 강제 새로고침
        await RefreshAsync();
        StatusText = "초기화 완료";
    }
}
