using ScrewMes.Core.Models;

namespace ScrewMes.Core.Abstractions;

/// <summary>나사 이미지 1장 추론. 백엔드(ONNX 네이티브 / REST)를 이 인터페이스로 추상화해
/// 상위 레이어(Data/App/Simulator)가 추론 구현에 의존하지 않게 한다.</summary>
public interface IScrewInferencer
{
    /// <param name="imagePath">원본 이미지 경로.</param>
    /// <param name="render">히트맵 PNG 생성 여부.</param>
    InspectionResult Predict(string imagePath, bool render = true);
}
