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

        if (!SupportedValueValidator.IsSupportedCoin(resolvedCoin))
        {
            throw new ArgumentException($"Unsupported coin: {resolvedCoin}");
        }

        if (resolvedAnalyzer != "all"
            && !SupportedValueValidator.IsSupportedAnalyzer(resolvedAnalyzer))
        {
            throw new ArgumentException(
                $"Unsupported analyzer: {resolvedAnalyzer}"
            );
        }

        if (!DateRangeValidator.IsValid(resolvedStartDate, resolvedEndDate))
        {
            throw new ArgumentException(
                "end_date must be on or after start_date"
            );
        }

        List<string>? resolvedSources = request.Source?
            .Select(source => source.ToLowerInvariant())
            .ToList();

        if (resolvedSources is not null
            && resolvedSources.Any(source =>
                !SupportedValueValidator.IsSupportedSource(source)))
        {
            throw new ArgumentException("Unsupported source");
        }

        int limit = request.Limit ?? 10;
        if (limit < 1)
        {
            throw new ArgumentException("limit must be at least 1");
        }

        IReadOnlyCollection<string> resolvedAnalyzers =
            resolvedAnalyzer == "all"
                ? SupportedValueValidator.GetSupportedAnalyzers()
                : [resolvedAnalyzer];

        var query =
            from sentiment in _database.Sentiments
            join post in _database.Posts
                on new
                {
                    sentiment.Coin,
                    sentiment.Source,
                    sentiment.ContentHash
                }
                equals new
                {
                    post.Coin,
                    post.Source,
                    post.ContentHash
                }
            where sentiment.Coin == resolvedCoin
            where resolvedAnalyzers.Contains(sentiment.Analyzer)
            where post.Timestamp >= resolvedStartDate
            where post.Timestamp <= resolvedEndDate
            select new
            {
                Sentiment = sentiment,
                Post = post
            };

        if (resolvedSources is { Count: > 0 })
        {
            query = query.Where(
                row => resolvedSources.Contains(row.Sentiment.Source));
        }

        List<SentimentResponse> matchingRows = await query
            .OrderBy(row => row.Post.Timestamp)
            .Select(row => new SentimentResponse
            {
                Coin = row.Sentiment.Coin,
                Source = row.Sentiment.Source,
                SourceId = row.Post.SourceId,
                Timestamp = row.Post.Timestamp,
                Text = row.Post.Text,
                Url = row.Post.Url,
                ContentHash = row.Sentiment.ContentHash,
                Analyzer = row.Sentiment.Analyzer,
                Sentiment = row.Sentiment.SentimentValue
            })
            .ToListAsync();

        return EvenlySample(matchingRows, limit);
    }

    private static List<SentimentResponse> EvenlySample(
        List<SentimentResponse> rows,
        int limit)
    {
        if (rows.Count <= limit)
        {
            return rows;
        }

        if (limit == 1)
        {
            return [rows[^1]];
        }

        return Enumerable.Range(0, limit)
            .Select(index =>
            {
                int rowIndex = (int)(
                    (long)index * (rows.Count - 1) / (limit - 1)
                );
                return rows[rowIndex];
            })
            .ToList();
    }
}
