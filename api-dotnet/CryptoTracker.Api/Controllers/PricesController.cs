using CryptoTracker.Api.Data;
using CryptoTracker.Api.Services;
using CryptoTracker.Api.Contracts.Requests;
using CryptoTracker.Api.Contracts.Responses;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;

namespace CryptoTracker.Api.Controllers;

[ApiController]
[Route("prices")]
public class PricesController : ControllerBase
{
    private readonly PriceService _priceService;

    public PricesController(PriceService priceService)
    {
        _priceService = priceService;
    }

    [HttpGet]
    public async Task<ActionResult<List<PriceResponse>>> GetPrices(
        [FromQuery] PriceQuery request)
    {
        try
        {
            List<PriceResponse> prices = 
                await _priceService.GetPricesAsync(request);

            return Ok(prices);
        }
        catch(ArgumentException exception)
        {
            return BadRequest(exception.Message);
        }
    }
}
