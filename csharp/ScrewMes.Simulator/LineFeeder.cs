using ScrewMes.Core;
using ScrewMes.Core.Abstractions;
using ScrewMes.Core.Entities;

namespace ScrewMes.Simulator;

/// <summary>라인 시뮬레이터 — test 이미지를 컨베이어처럼 1장씩 추론·기록 (Python line_feeder 포팅).
///
/// 실제 카메라 대신 data/screw/test 사진을 interval 간격으로 흘려보낸다. 각 장:
/// Predict(추론+히트맵) → InspectionRepository.Insert(현재 Lot에 귀속).
///
/// good_ratio: test셋은 결함 75%지만 실제 라인은 양품이 대부분이라, 같은 사진을 쓰되
/// '양품을 더 자주 뽑도록' 유입 빈도만 현실화한다(복원추출).
///
/// DefectType: 모델 산출이 아니라 소스 폴더명을 GT로 주입(Pareto용). good 소스면 null.
///
/// 주의: Predict는 동기·CPU 작업이므로 호출 측에서 Task.Run 등 백그라운드 스레드로 돌릴 것.</summary>
public sealed class LineFeeder
{
    private readonly IScrewInferencer _inferencer;
    private readonly IInspectionRepository _repository;
    private readonly string _feedSource;
    private readonly Random _rng = new();

    public LineFeeder(IScrewInferencer inferencer, IInspectionRepository repository, string feedSource)
    {
        _inferencer = inferencer;
        _repository = repository;
        _feedSource = feedSource;
    }

    /// <param name="lotId">검사를 귀속시킬 Lot.</param>
    /// <param name="count">처리 장수(null이면 취소될 때까지 무한).</param>
    /// <param name="interval">1장당 간격(초).</param>
    /// <param name="goodRatio">양품 유입 비율(0~1).</param>
    /// <param name="onInspected">각 검사 결과 콜백(UI 갱신용).</param>
    public async Task RunAsync(int lotId, int? count, double interval, double goodRatio,
        Action<Inspection>? onInspected, CancellationToken ct)
    {
        var (goods, bads) = SplitImages();
        if (goods.Count == 0 || bads.Count == 0)
            throw new InvalidOperationException(
                $"이미지 부족: good={goods.Count} bad={bads.Count} ({_feedSource})");

        var i = 0;
        while (!ct.IsCancellationRequested && (count is null || i < count))
        {
            i++;
            var pool = _rng.NextDouble() < goodRatio ? goods : bads;
            var path = pool[_rng.Next(pool.Count)];
            var folder = Directory.GetParent(path)!.Name;

            var r = _inferencer.Predict(path, render: true);
            var inspection = new Inspection
            {
                Ts = r.Ts,
                LotId = lotId,
                ImageId = r.ImageId,
                Verdict = r.Verdict,
                Score = r.Score,
                DefectType = folder == "good" ? null : folder,    // GT 유형 주입
                HeatmapPath = r.HeatmapPath,
            };
            await _repository.InsertAsync(inspection);
            onInspected?.Invoke(inspection);

            try { await Task.Delay(TimeSpan.FromSeconds(interval), ct); }
            catch (TaskCanceledException) { break; }
        }
    }

    /// <summary>test 이미지를 (양품, 불량) 리스트로 분리.</summary>
    private (List<string> goods, List<string> bads) SplitImages()
    {
        var goods = new List<string>();
        var bads = new List<string>();
        foreach (var dir in Directory.GetDirectories(_feedSource))
        {
            var isGood = Path.GetFileName(dir) == "good";
            foreach (var p in Directory.GetFiles(dir, "*.png"))
                (isGood ? goods : bads).Add(p);
        }
        return (goods, bads);
    }
}
