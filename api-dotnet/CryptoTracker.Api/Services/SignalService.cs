using CryptoTracker.Api.Contracts.Requests;
using CryptoTracker.Api.Contracts.Responses;
using CryptoTracker.Api.Data;
using CryptoTracker.Api.Validation;
using Microsoft.EntityFrameworkCore;

namespace CryptoTracker.Api.Services;

public class SignalService
{
    private readonly CryptoDbContext _database;

    public SignalService(CryptoDbContext database)
    {
        _database = database;
    }

    public async Task<List<SignalResponse>> GetSignalsAsync(SignalQuery request)
    {
        string resolvedCoin = string.IsNullOrWhiteSpace(request.Coin)
            ? "BTC"
            : request.Coin.ToUpperInvariant();
        List<string> resolvedSignalNames = request.SignalName is { Count: > 0 }
            ? request.SignalName
            : ["sma_20", "sma_50"];
        DateTimeOffset resolvedEndDate = request.EndDate ?? DateTimeOffset.UtcNow;
        DateTimeOffset resolvedStartDate =
            request.StartDate ?? resolvedEndDate.AddDays(-7);

        if (!SupportedValueValidator.IsSupportedCoin(resolvedCoin))
        {
            throw new ArgumentException($"Unsupported coin: {resolvedCoin}");
        }

        if (!DateRangeValidator.IsValid(resolvedStartDate, resolvedEndDate))
        {
            throw new ArgumentException(
                "end_date must be on or after start_date"
            );
        }

        int limit = Math.Clamp(request.NumSignals ?? 100, 1, 1000);

        return await _database.Signals
            .Where(signal => signal.Coin == resolvedCoin)
            .Where(signal => signal.Timestamp >= resolvedStartDate)
            .Where(signal => signal.Timestamp <= resolvedEndDate)
            .Where(signal => resolvedSignalNames.Contains(signal.SignalName))
            .OrderByDescending(signal => signal.Timestamp)
            .Take(limit)
            .Select(signal => new SignalResponse
            {
                Coin = signal.Coin,
                Timestamp = signal.Timestamp,
                SignalName = signal.SignalName,
                Value = signal.Value
            })
            .ToListAsync();
    }
}
