namespace CryptoTracker.Api.Contracts.Responses;

public class SentimentResponse
{
    public string Coin { get; init; } = "";
    public string Source { get; init; } = "";
    public string ContentHash { get; init; } = "";
    public string Analyzer { get; init; } = "";
    public double Sentiment { get; init; }
    public DateTimeOffset CreatedAt { get; init; }
}
