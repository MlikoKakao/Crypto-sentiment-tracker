using Microsoft.EntityFrameworkCore;

[Collection("Database")]
public class PriceEndpointTest : DatabaseTestCase
{
    public PriceEndpointTest(DatabaseFixture fixture)
        : base(fixture)
    {
    }
    [Fact]
    public async Task PriceTable_ContainsRows()
    {
        var prices = await DbContext.Prices.ToListAsync();

        Assert.NotEmpty(prices);
    }

    [Fact]
    public async Task PriceTable_ReturnsETH_()
    {
        var prices = await DbContext.Prices
            .Where(price => price.Coin == "ETH")
            .ToListAsync();

        Assert.NotEmpty(prices);
        Assert.All(prices, price => Assert.Equal("ETH", price.Coin));
    }
}