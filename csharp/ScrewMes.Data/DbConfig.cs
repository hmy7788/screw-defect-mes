namespace ScrewMes.Data;

/// <summary>DB 연결 설정. 현업 MES 표준에 맞춰 SQL Server(LocalDB) 사용.</summary>
public static class DbConfig
{
    /// <summary>로컬 개발용 LocalDB. 설치형 SQL Server면 Server= 만 바꾸면 된다.</summary>
    public const string ConnectionString =
        @"Server=(localdb)\MSSQLLocalDB;Database=ScrewMes;Trusted_Connection=True;TrustServerCertificate=True;";
}
