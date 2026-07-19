using CryptoTracker.Api.Models;

using Microsoft.EntityFrameworkCore;

namespace CryptoTracker.Api.Data;

public class CryptoDbContext : DbContext
{
    public CryptoDbContext(DbContextOptions<CryptoDbContext> options) : base(options)
    {
    }

    public DbSet<Price> Prices => Set<Price>();
    public DbSet<Sentiment> Sentiments => Set<Sentiment>();

    protected override void OnModelCreating(ModelBuilder modelBuilder)
    {
        modelBuilder.Entity<Price>(entity =>
                {
                    entity.ToTable("prices");

                    entity.HasKey(price => new
                    {
                        price.Coin,
                        price.Timestamp
                    });
                    entity.Property(price => price.Coin).HasColumnName("coin");

                    entity.Property(price => price.Timestamp).HasColumnName("timestamp");

                    entity.Property(price => price.PriceValue).HasColumnName("price");
                });

        modelBuilder.Entity<Sentiment>(entity =>
                {
                    entity.ToTable("sentiment");

                    entity.HasKey(Sentiment => new
                    {
                        Sentiment.Coin,
                        Sentiment.Source,
                        Sentiment.ContentHash,
                        Sentiment.Analyzer,
                    });
                    entity.Property(Sentiment => Sentiment.Coin).HasColumnName("coin");

                    entity.Property(Sentiment => Sentiment.Source).HasColumnName("source");

                    entity.Property(Sentiment => Sentiment.ContentHash).HasColumnName("content_hash");

                    entity.Property(Sentiment => Sentiment.Analyzer).HasColumnName("analyzer");

                    entity.Property(Sentiment => Sentiment.SentimentValue).HasColumnName("sentiment");

                    entity.Property(Sentiment => Sentiment.CreatedAt).HasColumnName("created_at");
                });

        modelBuilder.Entity<Post>(entity =>
                {
                    entity.ToTable("content_items");

                    entity.HasKey(Post => new
                    {
                        Post.Coin,
                        Post.Source,
                        Post.ContentHash
                    });
                    entity.Property(Post => Post.Coin).HasColumnName("coin");

                    entity.Property(Post => Post.Source).HasColumnName("source");

                    entity.Property(Post => Post.SourceId).HasColumnName("source_id");

                    entity.Property(Post => Post.Timestamp).HasColumnName("timestamp");

                    entity.Property(Post => Post.Text).HasColumnName("text");

                    entity.Property()
			    //TODO: Finish Post
                });
    }
}
