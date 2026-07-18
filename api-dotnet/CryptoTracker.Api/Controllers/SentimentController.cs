using CryptoTracker.Api.Data;
using CryptoTracker.Api.Models;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;

namespace CryptoTracker.Api.Controllers;

[ApiController]
[Route("Sentiment")]
public class SentimentController : ControllerBase
{
    private readonly CryptoDbContext _database;

    public SentimentController(CryptoDbContext database)
    {
        _database = database;
    }

    [HttpGet]
    public async Task<ActionResult<List<Sentiment>>> GetSentiment(
            [FromQuery]
            string? coin,
            DateTimeOffset? startDate,
            DateTimeOffset? endDate,
            List<string>? source,
            int? numSentiment
            )
    {
        IQueryable<Sentiment> query = _database.Sentiments;

        if (!string.IsNullOrWhiteSpace(coin))
        {
            query = query.Where(Sentiment => Sentiment.Coin == coin);
        }

        if (startDate.HasValue)
        {
            query = query.Where(Sentiment => Sentiment.CreatedAt >= startDate.Value);
        }
        if (endDate.HasValue)
        {
            query = query.Where(Sentiment => Sentiment.CreatedAt <= endDate.Value);
        }
        if (source is { Count: > 0})
        {
            query = query.Where(Sentiment => source.Contains(Sentiment.Source));
        }
        
        int limit = Math.Clamp(numSentiment ?? 100, 1, 1000);

        List<Sentiment> Sentiment = await query
            .OrderByDescending(Sentiment => Sentiment.CreatedAt)
            .Take(100)
            .ToListAsync();

        return Ok(Sentiment);
    }
}
