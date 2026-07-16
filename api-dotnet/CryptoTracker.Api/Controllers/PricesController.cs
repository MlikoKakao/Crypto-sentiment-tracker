using CryptoTracker.Api.Data;
using CryptoTracker.Api.Models;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;

namespace CryptoTracker.Api.Controllers;

[ApiController]
[Route("prices")]
public class PricesController : ControllerBase
{
    private readonly CryptoDbContext _database;

    public PricesController(CryptoDbContext database)
    {
        _database = database;
    }

    [HttpGet]
    public async Task<ActionResult<List<Price>>> GetPrices(
    [FromQuery] string? coin
)
    {
        IQueryable<Price> query = _database.Prices;

        if (!string.IsNullOrWhiteSpace(coin))
        {
            query = query.Where(price => price.Coin == coin);
        }

        List<Price> prices = await query
            .OrderByDescending(price => price.Timestamp)
            .Take(100)
            .ToListAsync();

        return Ok(prices);
    }
}
