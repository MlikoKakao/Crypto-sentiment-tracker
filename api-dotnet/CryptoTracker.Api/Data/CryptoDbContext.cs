using CryptoTracker.Api.Models;

using Microsoft.EntityFrameworkCore;

namespace CryptoTracker.Api.Data;

public class CryptoDbContext : DbContext
{
    public CryptoDbContext(DbContextOptions<CryptoDbContext> options) : base(options)
    {
    }

    public DbSet<Price> Prices => Set<Price>();

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

        modelBuilder.Entity<Post>(entity =>
                {
                    entity.ToTable("posts");

                    entity.HasKey(post => new
                    {
                        post.Coin,
                        post.Source,
                        post.ContentHash,
                        post.Analyzer,
                    });
                    entity.Property(post => post.Coin).HasColumnName("coin");

                    entity.Property(post => post.Source).HasColumnName("source");

                    entity.Property(post => post.ContentHash).HasColumnName("content_hash");

                    entity.Property(post => post.Analyzer).HasColumnName("analyzer");

                    entity.Property(post => post.Sentiment).HasColumnName("sentiment");

                    entity.Property(post => post.CreatedAt).HasColumnName("created_at");
                });
    }
}
