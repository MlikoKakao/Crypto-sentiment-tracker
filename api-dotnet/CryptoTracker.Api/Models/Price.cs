using System.ComponentModel.DataAnnotations.Schema;

namespace CryptoTracker.Api.Models;

public class Price
{
    [Column("coin")]
    public string Coin { get; set; } = "";

    [Column("timestamp")]
    public DateTimeOffset Timestamp { get; set; }

    [Column("price")]
    public double PriceValue { get; set; }
}
