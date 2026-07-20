using CryptoTracker.Api.Data;
using CryptoTracker.Api.Validation;
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
        [FromQuery(Name = "start_date")] DateTimeOffset? startDate,
        [FromQuery(Name = "end_date")] DateTimeOffset? endDate,
        [FromQuery] List<string>? signalName,
        [FromQuery] int? numSignals)
    {
        string resolvedCoin = string.IsNullOrWhiteSpace(coin)
            ? "BTC"
            : coin.ToUpperInvariant();
        List<string> resolvedSignalNames = signalName is { Count: > 0 }
            ? signalName
            : ["sma_20", "sma_50"];
        DateTimeOffset resolvedEndDate = endDate ?? DateTimeOffset.UtcNow;
        DateTimeOffset resolvedStartDate =
            startDate ?? resolvedEndDate.AddDays(-7);

        if (!DateRangeValidator.IsValid(resolvedStartDate, resolvedEndDate))
        {
            return BadRequest("end_date must be on or after start_date");
        }

        IQueryable<Signal> query = _database.Signals;

        query = query.Where(signal => signal.Coin == resolvedCoin);
        query = query.Where(signal => signal.Timestamp >= resolvedStartDate);
        query = query.Where(signal => signal.Timestamp <= resolvedEndDate);
        query = query.Where(
            signal => resolvedSignalNames.Contains(signal.SignalName));

        int limit = Math.Clamp(numSignals ?? 100, 1, 1000);

        List<Signal> signals = await query
            .OrderByDescending(signal => signal.Timestamp)
            .Take(limit)
            .ToListAsync();

        return Ok(signals);
    }
}
