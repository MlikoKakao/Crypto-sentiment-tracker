using System.Net;
using System.Net.Http.Json;
using CryptoTracker.Api.Contracts.Responses;

[Collection("Database")]
public class SignalEndpointTests : ApiDatabaseTestCase
{
    public SignalEndpointTests(DatabaseFixture fixture) : base(fixture)
    {
    }

    [Fact]
    public async Task GetSignals_ReturnsMatchingRows()
    {
        HttpResponseMessage response = await Client.GetAsync(
            "/signals?coin=BTC"
            + "&start_date=2026-07-20T00:00:00Z"
            + "&end_date=2026-07-21T00:00:00Z"
            + "&signalName=combined-sentiment"
        );
        List<SignalResponse>? signals =
            await response.Content.ReadFromJsonAsync<List<SignalResponse>>();

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.NotNull(signals);
        SignalResponse signal = Assert.Single(signals);
        Assert.Equal("BTC", signal.Coin);
        Assert.Equal("combined-sentiment", signal.SignalName);
        Assert.Equal(0.75, signal.Value);
    }

    [Fact]
    public async Task GetSignals_WithReversedDates_ReturnsBadRequest()
    {
        HttpResponseMessage response = await Client.GetAsync(
            "/signals?coin=BTC"
            + "&start_date=2026-07-21T00:00:00Z"
            + "&end_date=2026-07-20T00:00:00Z"
        );

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
    }
}
