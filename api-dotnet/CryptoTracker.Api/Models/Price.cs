using System.ComponentModel.DataAnnotations.Schema;

namespace CryptoTracker.Api.Models;

public class Price
{
    [Column("coin")]
    public string Coin { get; set; } = string.Empty;

    [Column("timestamp")]
    public DateTime Timestamp { get; set; }

    [Column("price")]
    public decimal PriceValue { get; set; }
}
