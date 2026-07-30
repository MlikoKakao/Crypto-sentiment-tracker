namespace CryptoTracker.Api.Contracts.Responses;

public class PriceResponse
{
    public string Coin { get; init; } = "";
    public DateTimeOffset Timestamp { get; init; }
    public double Price { get; init; }
}