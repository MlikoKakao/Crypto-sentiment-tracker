using Microsoft.AspNetCore.Mvc;

namespace CryptoTracker.Api.Contracts.Requests;

public class SignalQuery
{
    [FromQuery(Name = "coin")]
    public string? Coin { get; init; }

    [FromQuery(Name = "start_date")]
    public DateTimeOffset? StartDate { get; init; }

    [FromQuery(Name = "end_date")]
    public DateTimeOffset? EndDate { get; init; }

    [FromQuery(Name = "signalName")]
    public List<string>? SignalName { get; init; }

    [FromQuery(Name = "numSignals")]
    public int? NumSignals { get; init; }
}
