using Microsoft.EntityFrameworkCore;
using CryptoTracker.Api.Contracts.Requests;
using CryptoTracker.Api.Contracts.Responses;
using CryptoTracker.Api.Validation;
using CryptoTracker.Api.Data;

namespace CryptoTracker.Api.Services;

public class PriceService
{
    private readonly CryptoDbContext _database;

    public PriceService(CryptoDbContext database)
    {
        _database = database;
    }

    public async Task<List<PriceResponse>> GetPricesAsync(PriceQuery request)
    {
        string resolvedCoin = string.IsNullOrWhiteSpace(request.Coin)
            ? "BTC"
            : request.Coin.ToUpperInvariant();
        DateTimeOffset resolvedEndDate = request.EndDate ?? DateTimeOffset.UtcNow;
        DateTimeOffset resolvedStartDate =
            request.StartDate ?? resolvedEndDate.AddDays(-7);

        if (!DateRangeValidator.IsValid(resolvedStartDate, resolvedEndDate))
        {
            throw new ArgumentException(
                "end_date must be on or after start_date"
            );
        }

        List<PriceResponse> prices = await _database.Prices
            .Where(price => price.Coin == resolvedCoin)
            .Where(price => price.Timestamp >= resolvedStartDate)
            .Where(price => price.Timestamp <= resolvedEndDate)
            .OrderByDescending(price => price.Timestamp)
            .Select(price => new PriceResponse
            {
                Coin = price.Coin,
                Timestamp = price.Timestamp,
                Price = price.PriceValue
            })
            .ToListAsync();

        return prices;
    }
}