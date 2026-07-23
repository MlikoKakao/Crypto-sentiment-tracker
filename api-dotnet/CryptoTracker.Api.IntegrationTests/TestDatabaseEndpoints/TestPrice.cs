using Microsoft.AspNetCore.Mvc.Testing;
using System.Net;

public class PriceEndpointTest
{
    [Fact]
    public async Task GetHealth_ReturnHealthyStatus()
    {
        await using var application = new WebApplicationFactory<Program>();
        HttpClient client = application.CreateClient();

        HttpResponseMessage response = await client.GetAsync("/health");
        string body = await response.Content.ReadAsStringAsync();

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.Contains("healthy", body);
    }
}
