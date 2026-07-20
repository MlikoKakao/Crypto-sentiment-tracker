namespace CryptoTracker.Api.Contracts.Responses;

public class SignalResponse
{
    public string Coin { get; init; } = "";
    public DateTimeOffset Timestamp { get; init; }
    public string SignalName { get; init; } = "";
    public double Value { get; init; }
}
