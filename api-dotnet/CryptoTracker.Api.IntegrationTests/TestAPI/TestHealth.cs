using System.Net;
using System.Net.Http.Json;

[Collection("Database")]
public class HealthEndpointTests : ApiDatabaseTestCase
{
    public HealthEndpointTests(DatabaseFixture fixture) : base(fixture)
    {
    }

    [Fact]
    public async Task GetHealth_ReturnsHealthyStatus()
    {
        HttpResponseMessage response = await Client.GetAsync("/health");
        Dictionary<string, object>? body =
            await response.Content.ReadFromJsonAsync<Dictionary<string, object>>();

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.NotNull(body);
        Assert.Equal("healthy", body["status"].ToString());
    }
}
