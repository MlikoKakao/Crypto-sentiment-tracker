using CryptoTracker.Api.Contracts.Requests;
using CryptoTracker.Api.Contracts.Responses;
using CryptoTracker.Api.Data;
using CryptoTracker.Api.Validation;
using Microsoft.EntityFrameworkCore;

namespace CryptoTracker.Api.Services;

public class PostService
{
    private readonly CryptoDbContext _database;

    public PostService(CryptoDbContext database)
    {
        _database = database;
    }

    public async Task<List<PostResponse>> GetPostsAsync(PostQuery request)
    {
        string resolvedCoin = string.IsNullOrWhiteSpace(request.Coin)
            ? "BTC"
            : request.Coin.ToUpperInvariant();
        DateTimeOffset resolvedEndDate = request.EndDate ?? DateTimeOffset.UtcNow;
        DateTimeOffset resolvedStartDate =
            request.StartDate ?? resolvedEndDate.AddDays(-7);

        if (!SupportedValueValidator.IsSupportedCoin(resolvedCoin))
        {
            throw new ArgumentException($"Unsupported coin: {resolvedCoin}");
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

        var query = _database.Posts
            .Where(post => post.Coin == resolvedCoin)
            .Where(post => post.Timestamp >= resolvedStartDate)
            .Where(post => post.Timestamp <= resolvedEndDate);

        if (resolvedSources is { Count: > 0 })
        {
            query = query.Where(post => resolvedSources.Contains(post.Source));
        }

        int limit = request.NumPosts ?? 100;
        if (limit < 1)
        {
            throw new ArgumentException("numPosts must be at least 1");
        }

        return await query
            .OrderByDescending(post => post.Timestamp)
            .Take(limit)
            .Select(post => new PostResponse
            {
                Coin = post.Coin,
                Source = post.Source,
                SourceId = post.SourceId,
                Timestamp = post.Timestamp,
                Text = post.Text,
                Url = post.Url,
                ContentHash = post.ContentHash
            })
            .ToListAsync();
    }
}
