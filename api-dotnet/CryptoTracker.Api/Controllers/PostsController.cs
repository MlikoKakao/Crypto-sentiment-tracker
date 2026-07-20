using CryptoTracker.Api.Data;
using CryptoTracker.Api.Models;
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
        DateTimeOffset? startDate,
        DateTimeOffset? endDate,
        List<string>? source,
        int? numPosts)
    {
        IQueryable<Post> query = _database.Posts;

        if (!string.IsNullOrWhiteSpace(coin))
        {
            query = query.Where(post => post.Coin == coin);
        }

        if (startDate.HasValue)
        {
            query = query.Where(post => post.Timestamp >= startDate.Value);
        }

        if (endDate.HasValue)
        {
            query = query.Where(post => post.Timestamp <= endDate.Value);
        }

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
