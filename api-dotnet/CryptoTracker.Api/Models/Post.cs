
using System.ComponentModel.DataAnnotations.Schema;

namespace CryptoTracker.Api.Models;

public class Post
{

    [Column("coin")]
    public string Coin { get; set; } = "";

    [Column("source")]
    public string Source { get; set; } = "";

    [Column("source_id")]
    public string? SourceId { get; set; }

    [Column("timestamp")]
    public DateTimeOffset Timestamp { get; set; }

    [Column("text")]
    public string Text { get; set; } = "";

    [Column("url")]
    public string? Url { get; set; }

    [Column("content_hash")]
    public string ContentHash { get; set; } = "";
}
