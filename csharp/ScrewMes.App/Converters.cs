using System.Globalization;
using System.IO;
using System.Windows.Data;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using ScrewMes.Core;

namespace ScrewMes.App;

/// <summary>파일 경로 → 이미지 소스(파일 잠금 방지: OnLoad 캐싱).</summary>
public class PathToImageConverter : IValueConverter
{
    public object? Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        if (value is string path && !string.IsNullOrEmpty(path) && File.Exists(path))
        {
            var bmp = new BitmapImage();
            bmp.BeginInit();
            bmp.CacheOption = BitmapCacheOption.OnLoad;  // 파일 핸들 즉시 해제(라인 갱신 중 잠금 방지)
            bmp.UriSource = new Uri(path);
            bmp.EndInit();
            bmp.Freeze();
            return bmp;
        }
        return null;
    }

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        => throw new NotSupportedException();
}

/// <summary>판정 → 색 (Bad=적, Good=녹).</summary>
public class VerdictToBrushConverter : IValueConverter
{
    private static readonly SolidColorBrush Bad = new(Color.FromRgb(0xD3, 0x2F, 0x2F));
    private static readonly SolidColorBrush Good = new(Color.FromRgb(0x2E, 0x7D, 0x32));

    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        => value is Verdict v && v == Verdict.Bad ? Bad : Good;

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        => throw new NotSupportedException();
}

/// <summary>bool(불량 여부) → 배너 배경색.</summary>
public class BadToBannerBrushConverter : IValueConverter
{
    private static readonly SolidColorBrush Bad = new(Color.FromRgb(0xFD, 0xEC, 0xEA));
    private static readonly SolidColorBrush Good = new(Color.FromRgb(0xE8, 0xF5, 0xE9));

    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        => value is true ? Bad : Good;

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        => throw new NotSupportedException();
}
