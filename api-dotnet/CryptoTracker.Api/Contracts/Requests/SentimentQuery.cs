using Microsoft.AspNetCore.Mvc;

namespace CryptoTracker.Api.Contracts.Requests;

public class SentimentQuery
{
    [FromQuery(Name = "coin")]
    public string? Coin { get; init; }

    [FromQuery(Name = "start_date")]
    public DateTimeOffset? StartDate { get; init; }

    [FromQuery(Name = "end_date")]
    public DateTimeOffset? EndDate { get; init; }

    [FromQuery(Name = "source")]
    public List<string>? Source { get; init; }

    [FromQuery(Name = "analyzer")]
    public string? Analyzer { get; init; }

    [FromQuery(Name = "limit")]
    public int? Limit { get; init; }
}
