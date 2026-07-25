using CryptoTracker.Api.Data;
using Microsoft.EntityFrameworkCore;
using Npgsql;

public abstract class DatabaseTestCase : IDisposable
{
    protected DatabaseTestCase(DatabaseFixture databaseFixture)
    {
        var id = Guid.NewGuid().ToString().Replace("-", "");

        var databaseName = $"crypto_db_test_{id}";

        using (var tmplConnection = new NpgsqlConnection(databaseFixture.Connection))
        {
            tmplConnection.Open();

            using (var cmd = new NpgsqlCommand($"CREATE DATABASE {databaseName} WITH TEMPLATE {databaseFixture.TemplateDatabaseName}", tmplConnection))
            {
                cmd.ExecuteNonQuery();
            }
        }

        var connection = $"Host=localhost;Database={databaseName};Username=postgres;Password=postgres";

        var optionsBuilder = new DbContextOptionsBuilder<CryptoDbContext>();
        optionsBuilder.UseNpgsql(connection);

        DbContext = new CryptoDbContext(optionsBuilder.Options);
    }

    public CryptoDbContext DbContext { get; }

    public void Dispose()
    {
        DbContext.Database.EnsureDeleted();
    }
}