# ScrewMes — C# 산업형 MES (WPF · ONNX · SQL Server)

Python 연구 트랙에서 학습한 **PatchCore 이상탐지 모델**을, 현업 제조 IT에서 쓰는 형태의
**MES(제조실행시스템)** 로 운용하는 .NET 데스크톱 애플리케이션.

> 같은 모델의 두 운용 형태 중 "현업형" 버전. (연구·데모용 Python/Streamlit PoC는 `../mes/`)
> 외부 Python 프로세스 없이 **ONNX Runtime으로 .NET에서 직접 추론**한다.

## 아키텍처 (레이어드)

```
ScrewMes.Core        도메인 모델(WorkOrder/Lot/Inspection) + 인터페이스 (의존성 0)
ScrewMes.Inference   ONNX Runtime 추론 + 전처리 + jet 히트맵 (IScrewInferencer 구현)
ScrewMes.Data        EF Core + SQL Server(LocalDB), 리포지토리, 코드퍼스트 마이그레이션
ScrewMes.Simulator   라인 피더(test 이미지를 컨베이어처럼 흘려보냄)
ScrewMes.App         WPF(MVVM) — 생산현황 / 작업지시·추적성 / 품질분석 3탭
ScrewMes.ParityCheck C# 추론이 Python과 일치하는지 검증하는 콘솔
```

추론 백엔드는 `IScrewInferencer` 로 추상화 → ONNX 네이티브를 REST 등으로 교체해도 상위 무영향.

## 핵심: ONNX 네이티브 추론 파리티

anomalib `ExportType.ONNX` 산출물은 **전처리(Resize+Normalize)·후처리(점수 정규화+임계)를
그래프에 내장**한다. 따라서 C#은 원본 RGB를 `/255` 한 `[1,3,H,W]` 텐서만 넣으면
`pred_score`(정규화 이상점수)가 나오고, 여기에 임계값 0.513을 적용하면 끝이다.

검증(`ScrewMes.ParityCheck`): test 36장에서 **C# 점수 == Python predict_one** —
최대 |Δ| = **0.00010**(최초 학습 모델; 재학습 모델은 0.0016), 판정 일치 **36/36**. (전처리 형태는 export 시 실측해
`model/meta.json`에 기록 → C#이 그대로 따른다.)

## 사전 요구

1. **.NET 8 SDK** — https://aka.ms/dotnet/download (`winget install Microsoft.DotNet.SDK.8`)
2. **SQL Server LocalDB** — 관리자 권한으로 1회 설치
   (SqlLocalDB.msi: `msiexec /i SqlLocalDB.msi IACCEPTSQLLOCALDBLICENSETERMS=YES`)
   설치형 SQL Server를 쓰면 `ScrewMes.Data/DbConfig.cs`의 연결문자열만 바꾼다.
3. **모델 export** (레포 루트, conda `screw_defect`):
   ```bash
   python -m mes.core.export_model      # (최초 1회) ckpt → pt.  (파리티 정답 산출에 사용)
   python -m mes.core.export_onnx       # ckpt → csharp/model/model.onnx + meta.json + parity_samples.json
   ```

## 빌드 · 실행

```bash
cd csharp
dotnet build ScrewMes.sln

# (선택) C# 추론이 Python과 일치하는지 검증
dotnet run --project ScrewMes.ParityCheck -c Release

# MES 실행 — 첫 실행 시 LocalDB에 스키마 자동 생성(Migrate)
dotnet run --project ScrewMes.App -c Release
```

앱에서: 품목·계획수량 입력 → **▶ 가동** → 생산현황이 실시간 갱신(KPI·추세·히트맵 갤러리),
**작업지시·추적성** 탭에서 WO→Lot→검사 드릴다운, **품질분석** 탭에서 불량 유형 Pareto.

## 데이터 모델 / 정직성 노트

- `WorkOrder` 1—N `Lot` 1—N `Inspection` (추적성 genealogy).
- `Inspection.DefectType` 은 **모델 산출이 아니다**. PatchCore는 good/bad만 판정하므로,
  라인 시뮬레이터가 **소스 폴더명을 GT 유형으로 주입**해 Pareto를 구성한다(현업에선 작업자
  재검 시 입력하는 자리). good 소스를 불량 판정하면 `정상(오탐)`으로 집계된다.

## 검증 요약

| 항목 | 결과 |
|------|------|
| 솔루션 빌드 | 0 errors |
| C# ONNX 추론 vs Python | 최대 Δ 0.00010, 판정 36/36 일치 |
| 히트맵 렌더 | PNG 생성 확인(결함 위치 부각) |
| SQL Server DDL | 마이그레이션 유효(PK/FK cascade/인덱스) |
