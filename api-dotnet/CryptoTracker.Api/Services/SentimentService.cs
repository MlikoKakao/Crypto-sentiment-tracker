using CryptoTracker.Api.Contracts.Requests;
using CryptoTracker.Api.Contracts.Responses;
using CryptoTracker.Api.Data;
using CryptoTracker.Api.Validation;
using Microsoft.EntityFrameworkCore;

namespace CryptoTracker.Api.Services;

public class SentimentService
{
    private readonly CryptoDbContext _database;

    public SentimentService(CryptoDbContext database)
    {
        _database = database;
    }

    public async Task<List<SentimentResponse>> GetSentimentAsync(
        SentimentQuery request)
    {
        string resolvedCoin = string.IsNullOrWhiteSpace(request.Coin)
            ? "BTC"
            : request.Coin.ToUpperInvariant();
        string resolvedAnalyzer = string.IsNullOrWhiteSpace(request.Analyzer)
            ? "vader"
            : request.Analyzer.ToLowerInvariant();
        DateTimeOffset resolvedEndDate = request.EndDate ?? DateTimeOffset.UtcNow;
        DateTimeOffset resolvedStartDate =
            request.StartDate ?? resolvedEndDate.AddDays(-7);

        if (!DateRangeValidator.IsValid(resolvedStartDate, resolvedEndDate))
        {
            throw new ArgumentException(
                "end_date must be on or after start_date"
            );
        }

        var query = _database.Sentiments
            .Where(sentiment => sentiment.Coin == resolvedCoin)
            .Where(sentiment => sentiment.Analyzer == resolvedAnalyzer)
            .Where(sentiment => sentiment.CreatedAt >= resolvedStartDate)
            .Where(sentiment => sentiment.CreatedAt <= resolvedEndDate);

        if (request.Source is { Count: > 0 })
        {
            query = query.Where(
                sentiment => request.Source.Contains(sentiment.Source));
        }

        int limit = Math.Clamp(request.NumSentiment ?? 100, 1, 1000);

        return await query
            .OrderByDescending(sentiment => sentiment.CreatedAt)
            .Take(limit)
            .Select(sentiment => new SentimentResponse
            {
                Coin = sentiment.Coin,
                Source = sentiment.Source,
                ContentHash = sentiment.ContentHash,
                Analyzer = sentiment.Analyzer,
                Sentiment = sentiment.SentimentValue,
                CreatedAt = sentiment.CreatedAt
            })
            .ToListAsync();
    }
}
