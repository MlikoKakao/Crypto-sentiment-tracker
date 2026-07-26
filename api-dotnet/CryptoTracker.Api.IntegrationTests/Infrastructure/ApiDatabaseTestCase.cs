using CryptoTracker.Api.Data;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.Configuration;

public abstract class ApiDatabaseTestCase : DatabaseTestCase
{
    private readonly WebApplicationFactory<Program> _application;
    protected HttpClient Client { get; }

    protected ApiDatabaseTestCase(DatabaseFixture fixture) : base(fixture)
    {
        _application = new WebApplicationFactory<Program>()
            .WithWebHostBuilder(builder =>
        {
            builder.ConfigureServices(services =>
            {
                ServiceDescriptor? existingOptions =
                    services.SingleOrDefault(service =>
                        service.ServiceType ==
                        typeof(DbContextOptions<CryptoDbContext>));

                if (existingOptions is not null)
                {
                    services.Remove(existingOptions);
                }

                services.AddDbContext<CryptoDbContext>(options =>
                    options.UseNpgsql(Connection));
            });
        });
        Client = _application.CreateClient();
    }

    public override void Dispose()
    {
        Client.Dispose();
        _application.Dispose();
        base.Dispose();
    }
}