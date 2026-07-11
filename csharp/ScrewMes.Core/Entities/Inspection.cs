namespace ScrewMes.Core.Entities;

/// <summary>검사 1건 = 한 행. 대시보드의 모든 지표가 이 테이블에서 나온다.</summary>
public class Inspection
{
    public int Id { get; set; }

    public DateTime Ts { get; set; } = DateTime.Now;

    public int LotId { get; set; }
    public Lot? Lot { get; set; }

    /// <summary>원본 이미지 식별자 (폴더_파일명).</summary>
    public string ImageId { get; set; } = "";

    public Verdict Verdict { get; set; }

    /// <summary>이상점수 (정규화, 0~1). 임계값 0.513 이상이면 Bad.</summary>
    public double Score { get; set; }

    /// <summary>결함 유형 — 모델 산출이 아니라 라인 시뮬레이터가 소스 폴더명으로 주입한 GT.
    /// (PatchCore는 good/bad만 판정; 현업에선 작업자 재검 시 입력하는 자리.) good 소스면 null.</summary>
    public string? DefectType { get; set; }

    /// <summary>결함 위치 히트맵 PNG 경로 (good은 null일 수 있음).</summary>
    public string? HeatmapPath { get; set; }
}
