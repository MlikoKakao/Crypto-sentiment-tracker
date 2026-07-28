using CryptoTracker.Api.Data;
using CryptoTracker.Api.Models;
using Microsoft.EntityFrameworkCore;

public class DatabaseFixture : IDisposable
{
    private readonly CryptoDbContext _context;

    public DatabaseFixture()
    {
        string id = Guid.NewGuid().ToString().Replace("-", "");

        TemplateDatabaseName = $"crypto_db_test_{id}";

        Connection = $"Host=localhost;Database={TemplateDatabaseName};Username=postgres;Password=postgres";

        var optionsBuilder = new DbContextOptionsBuilder<CryptoDbContext>();
        optionsBuilder.UseNpgsql(Connection);

        _context = new CryptoDbContext(optionsBuilder.Options);
    
        _context.Database.EnsureCreated();

        var firstTimestamp =
            new DateTimeOffset(2026, 7, 20, 0, 0, 0, TimeSpan.Zero);
        var secondTimestamp = firstTimestamp.AddHours(1);

        _context.Prices.AddRange(
            new Price
            {
                Coin = "BTC",
                Timestamp = firstTimestamp,
                PriceValue = 118_000
            },
            new Price
            {
                Coin = "ETH",
                Timestamp = secondTimestamp,
                PriceValue = 3_600
            }
        );

        _context.Posts.Add(new Post
        {
            Coin = "BTC",
            Source = "reddit",
            SourceId = "test-post-1",
            Timestamp = firstTimestamp,
            Text = "Bitcoin sentiment is positive today.",
            Url = "https://example.com/test-post-1",
            ContentHash = "test-content-hash-1"
        });

        _context.Sentiments.AddRange(
            new Sentiment
            {
                Coin = "BTC",
                Source = "reddit",
                ContentHash = "test-content-hash-1",
                Analyzer = "vader",
                SentimentValue = 0.75,
                CreatedAt = secondTimestamp.AddDays(10)
            },
            new Sentiment
            {
                Coin = "BTC",
                Source = "reddit",
                ContentHash = "test-content-hash-1",
                Analyzer = "textblob",
                SentimentValue = 0.50,
                CreatedAt = secondTimestamp.AddDays(10)
            }
        );

        _context.Signals.Add(new Signal
        {
            Coin = "BTC",
            Timestamp = secondTimestamp,
            SignalName = "combined-sentiment",
            Value = 0.75
        });

        _context.SaveChanges();

        _context.Database.CloseConnection();
    }

    public string TemplateDatabaseName { get; }

    public string Connection { get; }

    public void Dispose()
        {
            _context.Database.EnsureDeleted();
        }
    
}
