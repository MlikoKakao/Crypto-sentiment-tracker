using System.Net;
using System.Net.Http.Json;
using CryptoTracker.Api.Contracts.Responses;

[Collection("Database")]
public class PriceEndpointTests : ApiDatabaseTestCase
{
    public PriceEndpointTests(DatabaseFixture fixture) : base(fixture)
    {
    }

    [Fact]
    public async Task GetPrices_ReturnsMatchingRows()
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
        Assert.Equal(3_600, prices[0].Price);
    }

    [Fact]
    public async Task GetPrices_WithReversedDates_ReturnsBadRequest()
    {
        HttpResponseMessage response = await Client.GetAsync(
            "/prices?coin=BTC"
            + "&start_date=2026-07-21T00:00:00Z"
            + "&end_date=2026-07-20T00:00:00Z"
        );
        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
    }

    [Fact]
    public async Task GetPrices_WithNoMatchingRows_ReturnsEmptyArray()
    {
        HttpResponseMessage response = await Client.GetAsync(
            "/prices?coin=XMR"
            + "&start_date=2026-07-20T00:00:00Z"
            + "&end_date=2026-07-21T00:00:00Z"
        );
        List<PriceResponse>? prices =
            await response.Content.ReadFromJsonAsync<List<PriceResponse>>();

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.NotNull(prices);
        Assert.Empty(prices);
    }
}
