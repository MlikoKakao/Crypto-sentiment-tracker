using System;
using CryptoTracker.Api.Data;
using Microsoft.EntityFrameworkCore;

public class DatabaseFixture : IDisposable
{
    private readonly CryptoDbContext _context;

    public DatabaseFixture()
    {
        string id = Guid.NewGuid().ToString().Replace("-", "");

        TemplateDatabaseName = $"crypto_db_test_{id}";

        Connection = $"Host=localhost;Database={TemplateDatabaseName};Username=postgre_test;Password=postgre_test";

        var optionsBuilder = new DbContextOptionsBuilder<CryptoDbContext>();
        optionsBuilder.UseNpgsql(Connection);

        _context = new CryptoDbContext(optionsBuilder.Options);
    
        _context.Database.EnsureCreated();





        _context.Database.CloseConnection();
    }

    public string TemplateDatabaseName { get; }

    public string Connection { get; }

    public void Dispose()
        {
            _context.Database.EnsureDeleted();
        }
    
}
