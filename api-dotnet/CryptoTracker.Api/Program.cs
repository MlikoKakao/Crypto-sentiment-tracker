using CryptoTracker.Api.Data;
using Microsoft.EntityFrameworkCore;

var builder = WebApplication.CreateBuilder(args);

var connectionString =
    builder.Configuration.GetConnectionString("Postgres")
    ?? throw new InvalidOperationException(
        "Connection string 'Postgres' was not found.");

builder.Services.AddDbContext<CryptoDbContext>(options =>
{
    options.UseNpgsql(connectionString);
});

builder.Services.AddControllers();

var app = builder.Build();

app.MapControllers();

app.MapGet("/", () =>
{
    return Results.Ok(new
    {
        service = "CryptoTracker .NET API",
        status = "running"
    });
});

app.MapGet("/health", () =>
        {
            return Results.Ok(new
            {
                status = "healthy",
                timestamp = DateTime.UtcNow
            });
        });

app.Run();
