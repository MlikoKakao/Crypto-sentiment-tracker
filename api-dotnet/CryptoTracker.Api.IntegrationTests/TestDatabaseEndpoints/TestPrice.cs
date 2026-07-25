using Microsoft.EntityFrameworkCore;

[Collection("Database")]
public class PriceEndpointTest : DatabaseTestCase
{
    public PriceEndpointTest(DatabaseFixture fixture)
        : base(fixture)
    {
    }
        [Fact]
        public async Task PriceTable_ContainsBitcoin()
        {
            var prices = await DbContext.Prices.ToListAsync();

            Assert.NotEmpty(prices);
            Assert.Contains(prices, price => price.Coin == "BTC");
        }
}