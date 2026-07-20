using CryptoTracker.Api.Data;
using CryptoTracker.Api.Models;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;

namespace CryptoTracker.Api.Controllers;

[ApiController]
[Route("signals")]
public class SignalsController : ControllerBase
{
    private readonly CryptoDbContext _database;

    public SignalsController(CryptoDbContext database)
    {
        _database = database;
    }

    [HttpGet]
    public async Task<ActionResult<List<Signal>>> GetSignals(
        [FromQuery] string? coin,
        DateTimeOffset? startDate,
        DateTimeOffset? endDate,
        List<string>? signalName,
        int? numSignals)
    {
        IQueryable<Signal> query = _database.Signals;

        if (!string.IsNullOrWhiteSpace(coin))
        {
            query = query.Where(signal => signal.Coin == coin);
        }

        if (startDate.HasValue)
        {
            query = query.Where(signal => signal.Timestamp >= startDate.Value);
        }

        if (endDate.HasValue)
        {
            query = query.Where(signal => signal.Timestamp <= endDate.Value);
        }

        if (signalName is { Count: > 0 })
        {
            query = query.Where(signal => signalName.Contains(signal.SignalName));
        }

        int limit = Math.Clamp(numSignals ?? 100, 1, 1000);

        List<Signal> signals = await query
            .OrderByDescending(signal => signal.Timestamp)
            .Take(limit)
            .ToListAsync();

        return Ok(signals);
    }
}
