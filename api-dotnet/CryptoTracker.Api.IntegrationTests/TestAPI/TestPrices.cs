using System.Net;
using System.Net.Http.Json;
using CryptoTracker.Api.Contracts.Responses;
using Microsoft.AspNetCore.Http;

[Collection("Database")]
public class PriceEndpointTests : ApiDatabaseTestCase
{
    public PriceEndpointTests(DatabaseFixture fixture) : base(fixture)
    {
    }
    [Fact]
    public async Task TestAPI_ReturnsRows()
    {
        HttpResponseMessage response = await Client.GetAsync(
            "/prices?coin=ETH"
            + "&start_date=2026-07-20T00:00:00Z"
            + "&end_date=2026-07-21T00:00:00Z"
        );
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        List<PriceResponse>? prices =
            await response.Content.ReadFromJsonAsync<List<PriceResponse>>();

        Assert.NotNull(prices);
        Assert.NotEmpty(prices);
        Assert.All(prices, price => Assert.Equal("ETH", price.Coin));
    }

    [Fact]
    public async Task TestAPI_FailsReversedDates()
    {
        HttpResponseMessage response = await Client.GetAsync(
            "/prices?coin=BTC"
            + "&start_date=2026-07-21T00:00:00Z"
            + "&end_date=2026-07-20T00:00:00Z"
        );
        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
    }

}