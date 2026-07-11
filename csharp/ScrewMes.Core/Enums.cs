namespace ScrewMes.Core;

/// <summary>검사 판정. PatchCore 이상점수(정규화)에 임계값(0.513)을 적용한 결과.</summary>
public enum Verdict
{
    Good,
    Bad,
}

/// <summary>작업지시(Work Order) 진행 상태.</summary>
public enum OrderStatus
{
    Open,       // 생성됨, 아직 미착수
    Running,    // 생산 중
    Closed,     // 종료
}

/// <summary>Lot 진행 상태.</summary>
public enum LotStatus
{
    Open,
    Closed,
}
