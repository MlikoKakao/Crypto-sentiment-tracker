using CryptoTracker.Api.Data;
using CryptoTracker.Api.Models;
using CryptoTracker.Api.Validation;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;

namespace CryptoTracker.Api.Controllers;

[ApiController]
[Route("posts")]
public class PostsController : ControllerBase
{
    private readonly CryptoDbContext _database;

    public PostsController(CryptoDbContext database)
    {
        _database = database;
    }

    [HttpGet]
    public async Task<ActionResult<List<Post>>> GetPosts(
        [FromQuery] string? coin,
        [FromQuery(Name = "start_date")] DateTimeOffset? startDate,
        [FromQuery(Name = "end_date")] DateTimeOffset? endDate,
        [FromQuery] List<string>? source,
        [FromQuery] int? numPosts)
    {
        string resolvedCoin = string.IsNullOrWhiteSpace(coin)
            ? "BTC"
            : coin.ToUpperInvariant();
        DateTimeOffset resolvedEndDate = endDate ?? DateTimeOffset.UtcNow;
        DateTimeOffset resolvedStartDate =
            startDate ?? resolvedEndDate.AddDays(-7);

        if (!DateRangeValidator.IsValid(resolvedStartDate, resolvedEndDate))
        {
            return BadRequest("end_date must be on or after start_date");
        }

        IQueryable<Post> query = _database.Posts;

        query = query.Where(post => post.Coin == resolvedCoin);
        query = query.Where(post => post.Timestamp >= resolvedStartDate);
        query = query.Where(post => post.Timestamp <= resolvedEndDate);

        if (source is { Count: > 0 })
        {
            query = query.Where(post => source.Contains(post.Source));
        }

        int limit = Math.Clamp(numPosts ?? 100, 1, 1000);

        List<Post> posts = await query
            .OrderByDescending(post => post.Timestamp)
            .Take(limit)
            .ToListAsync();

        return Ok(posts);
    }
}
