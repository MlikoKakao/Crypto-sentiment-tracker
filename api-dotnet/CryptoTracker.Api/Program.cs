var builder = WebApplication.CreateBuilder(args);

builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

var app = builder.Build();

if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

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
