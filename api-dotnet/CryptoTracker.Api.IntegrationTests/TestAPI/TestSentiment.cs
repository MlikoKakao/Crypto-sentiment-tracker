    using System.Net;
using System.Net.Http.Json;
using CryptoTracker.Api.Contracts.Responses;

[Collection("Database")]
public class SentimentEndpointTests : ApiDatabaseTestCase
{
    public SentimentEndpointTests(DatabaseFixture fixture) : base(fixture)
    {
    }

    [Fact]
    public async Task GetSentiment_ReturnsMatchingRows()
    {
        HttpResponseMessage response = await Client.GetAsync(
            "/sentiment?coin=BTC"
            + "&start_date=2026-07-20T00:00:00Z"
            + "&end_date=2026-07-21T00:00:00Z"
            + "&source=reddit"
            + "&analyzer=vader"
        );
        List<SentimentResponse>? sentiments =
            await response.Content.ReadFromJsonAsync<List<SentimentResponse>>();

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.NotNull(sentiments);
        SentimentResponse sentiment = Assert.Single(sentiments);
        Assert.Equal("BTC", sentiment.Coin);
        Assert.Equal("reddit", sentiment.Source);
        Assert.Equal("vader", sentiment.Analyzer);
        Assert.Equal(0.75, sentiment.Sentiment);
    }

    [Fact]
    public async Task GetSentiment_WithInvalidCoin_ReturnsBadRequest()
    {
        HttpResponseMessage response = await Client.GetAsync(
            "/sentiment?coin=DOGE"
            + "&start_date=2026-07-20T00:00:00Z"
            + "&end_date=2026-07-21T00:00:00Z"
            + "&source=reddit"
        );

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
    }

    [Fact]
    public async Task GetSentiment_WithAnalyzerAll_CombinesAnalyzers()
    {
        HttpResponseMessage response = await Client.GetAsync(
            "/sentiment?coin=BTC"
            + "&start_date=2026-07-20T00:00:00Z"
            + "&end_date=2026-07-21T00:00:00Z"
            + "&source=reddit"
            + "&analyzer=all"
        );
        List<SentimentResponse>? sentiments =
            await response.Content.ReadFromJsonAsync<List<SentimentResponse>>();

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.NotNull(sentiments);
        Assert.Equal(
            ["textblob", "vader"],
            sentiments.Select(row => row.Analyzer).Order().ToList()
        );
    }
}
