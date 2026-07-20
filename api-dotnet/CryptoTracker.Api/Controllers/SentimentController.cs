using CryptoTracker.Api.Data;
using CryptoTracker.Api.Validation;
using CryptoTracker.Api.Models;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;

namespace CryptoTracker.Api.Controllers;

[ApiController]
[Route("sentiment")]
public class SentimentController : ControllerBase
{
    private readonly CryptoDbContext _database;

    public SentimentController(CryptoDbContext database)
    {
        _database = database;
    }

    [HttpGet]
    public async Task<ActionResult<List<Sentiment>>> GetSentiment(
        [FromQuery] string? coin,
        [FromQuery(Name = "start_date")] DateTimeOffset? startDate,
        [FromQuery(Name = "end_date")] DateTimeOffset? endDate,
        [FromQuery] List<string>? source,
        [FromQuery] string? analyzer,
        [FromQuery] int? numSentiment)
    {
        string resolvedCoin = string.IsNullOrWhiteSpace(coin)
            ? "BTC"
            : coin.ToUpperInvariant();
        string resolvedAnalyzer = string.IsNullOrWhiteSpace(analyzer)
            ? "vader"
            : analyzer.ToLowerInvariant();
        DateTimeOffset resolvedEndDate = endDate ?? DateTimeOffset.UtcNow;
        DateTimeOffset resolvedStartDate =
            startDate ?? resolvedEndDate.AddDays(-7);

        if (!DateRangeValidator.IsValid(resolvedStartDate, resolvedEndDate))
        {
            return BadRequest("end_date must be on or after start_date");
        }

        IQueryable<Sentiment> query = _database.Sentiments;

        query = query.Where(sentiment => sentiment.Coin == resolvedCoin);
        query = query.Where(sentiment => sentiment.Analyzer == resolvedAnalyzer);
        query = query.Where(
            sentiment => sentiment.CreatedAt >= resolvedStartDate);
        query = query.Where(
            sentiment => sentiment.CreatedAt <= resolvedEndDate);

        if (source is { Count: > 0 })
        {
            query = query.Where(sentiment => source.Contains(sentiment.Source));
        }

        int limit = Math.Clamp(numSentiment ?? 100, 1, 1000);

        List<Sentiment> sentiments = await query
            .OrderByDescending(sentiment => sentiment.CreatedAt)
            .Take(limit)
            .ToListAsync();

        return Ok(sentiments);
    }
}
