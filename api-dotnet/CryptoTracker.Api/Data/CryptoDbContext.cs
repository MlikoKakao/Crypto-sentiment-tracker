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
    }
}
