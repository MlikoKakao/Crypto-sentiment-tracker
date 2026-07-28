using System.Net;
using System.Net.Http.Json;
using CryptoTracker.Api.Contracts.Responses;
using CryptoTracker.Api.Models;

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
        Assert.Equal("test-post-1", sentiment.SourceId);
        Assert.Equal(
            new DateTimeOffset(2026, 7, 20, 0, 0, 0, TimeSpan.Zero),
            sentiment.Timestamp
        );
        Assert.Equal("Bitcoin sentiment is positive today.", sentiment.Text);
        Assert.Equal("https://example.com/test-post-1", sentiment.Url);
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

    [Fact]
    public async Task GetSentiment_WithInvalidAnalyzer_ReturnsBadRequest()
    {
        HttpResponseMessage response = await Client.GetAsync(
            "/sentiment?coin=BTC"
            + "&start_date=2026-07-20T00:00:00Z"
            + "&end_date=2026-07-21T00:00:00Z"
            + "&source=reddit"
            + "&analyzer=unsupported"
        );

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1001)]
    public async Task GetSentiment_WithInvalidLimit_ReturnsBadRequest(int limit)
    {
        HttpResponseMessage response = await Client.GetAsync(
            "/sentiment?coin=BTC"
            + "&start_date=2026-07-20T00:00:00Z"
            + "&end_date=2026-07-21T00:00:00Z"
            + $"&limit={limit}"
        );

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
    }

    [Fact]
    public async Task GetSentiment_WithInvalidSource_ReturnsBadRequest()
    {
        HttpResponseMessage response = await Client.GetAsync(
            "/sentiment?coin=BTC"
            + "&start_date=2026-07-20T00:00:00Z"
            + "&end_date=2026-07-21T00:00:00Z"
            + "&source=unsupported"
        );

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
    }

    [Fact]
    public async Task GetSentiment_WithoutLimit_ReturnsAtMostTenRows()
    {
        DateTimeOffset timestamp =
            new(2026, 7, 20, 2, 0, 0, TimeSpan.Zero);

        for (int index = 0; index < 10; index++)
        {
            string contentHash = $"limit-test-{index}";
            DbContext.Posts.Add(new Post
            {
                Coin = "BTC",
                Source = "reddit",
                Timestamp = timestamp.AddMinutes(index),
                Text = $"Limit test post {index}",
                ContentHash = contentHash
            });
            DbContext.Sentiments.Add(new Sentiment
            {
                Coin = "BTC",
                Source = "reddit",
                ContentHash = contentHash,
                Analyzer = "vader",
                SentimentValue = 0.1,
                CreatedAt = timestamp.AddDays(10)
            });
        }

        await DbContext.SaveChangesAsync();

        HttpResponseMessage response = await Client.GetAsync(
            "/sentiment?coin=BTC"
            + "&start_date=2026-07-20T00:00:00Z"
            + "&end_date=2026-07-21T00:00:00Z"
        );
        List<SentimentResponse>? sentiments =
            await response.Content.ReadFromJsonAsync<List<SentimentResponse>>();

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.NotNull(sentiments);
        Assert.Equal(10, sentiments.Count);
    }

    [Fact]
    public async Task GetSentiment_WithoutParameters_UsesDefaults()
    {
        HttpResponseMessage response = await Client.GetAsync("/sentiment");

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
    }
}
