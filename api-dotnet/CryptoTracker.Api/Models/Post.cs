
using System.ComponentModel.DataAnnotations.Schema;

namespace CryptoTracker.Api.Models;

public class Post
{

    [Column("coin")]
    public string Coin { get; set; } = "";

    [Column("source")]
    public string Source { get; set; } = "";

    [Column("content_hash")]
    public string ContentHash { get; set; } = "";

    [Column("analyzer")]
    public string Analyzer { get; set; } = "";

    [Column("sentiment")]
    public decimal Sentiment { get; set; }
}
