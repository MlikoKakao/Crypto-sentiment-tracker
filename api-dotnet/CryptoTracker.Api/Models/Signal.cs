using System.ComponentModel.DataAnnotations.Schema;

namespace CryptoTracker.Api.Models;

public class Signal
{
    [Column("coin")]
    public string Coin { get; set; } = "";

    [Column("timestamp")]
    public DateTimeOffset Timestamp { get; set; }

    [Column("signal_name")]
    public string SignalName { get; set; } = "";

    [Column("value")]
    public double Value { get; set; }
}
