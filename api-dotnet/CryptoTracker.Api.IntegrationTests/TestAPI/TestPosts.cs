using System.Net;
using System.Net.Http.Json;
using CryptoTracker.Api.Contracts.Responses;

[Collection("Database")]
public class PostEndpointTests : ApiDatabaseTestCase
{
    public PostEndpointTests(DatabaseFixture fixture) : base(fixture)
    {
    }

    [Fact]
    public async Task GetPosts_ReturnsMatchingRows()
    {
        HttpResponseMessage response = await Client.GetAsync(
            "/posts?coin=BTC"
            + "&start_date=2026-07-20T00:00:00Z"
            + "&end_date=2026-07-21T00:00:00Z"
            + "&source=reddit"
        );
        List<PostResponse>? posts =
            await response.Content.ReadFromJsonAsync<List<PostResponse>>();

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.NotNull(posts);
        PostResponse post = Assert.Single(posts);
        Assert.Equal("BTC", post.Coin);
        Assert.Equal("reddit", post.Source);
        Assert.Equal("test-content-hash-1", post.ContentHash);
    }

    [Fact]
    public async Task GetPosts_WithReversedDates_ReturnsBadRequest()
    {
        HttpResponseMessage response = await Client.GetAsync(
            "/posts?coin=BTC"
            + "&start_date=2026-07-21T00:00:00Z"
            + "&end_date=2026-07-20T00:00:00Z"
        );

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
    }

    [Fact]
    public async Task GetPosts_WithInvalidSource_ReturnsBadRequest()
    {
        HttpResponseMessage response = await Client.GetAsync(
            "/posts?coin=BTC"
            + "&start_date=2026-07-20T00:00:00Z"
            + "&end_date=2026-07-21T00:00:00Z"
            + "&source=unsupported"
        );

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
    }

    [Fact]
    public async Task GetPosts_WithoutParameters_UsesDefaults()
    {
        HttpResponseMessage response = await Client.GetAsync("/posts");

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
    }
}
