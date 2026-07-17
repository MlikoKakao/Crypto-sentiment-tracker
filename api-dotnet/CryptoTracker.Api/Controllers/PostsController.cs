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
            [FromQuery]
            string? coin,
            DateTimeOffset? StartDate,
            DateTimeOffset? EndDate,
            // TODO: add list[Source]
            int? NumPosts
            )
    {
        IQueryable<Post> query = _database.Posts;

        if (!string.IsNullOrWhiteSpace(coin))
        {
            query = query.Where(post => post.Coin == coin);
        }
    }
}
