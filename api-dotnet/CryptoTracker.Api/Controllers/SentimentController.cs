using CryptoTracker.Api.Contracts.Requests;
using CryptoTracker.Api.Contracts.Responses;
using CryptoTracker.Api.Services;
using Microsoft.AspNetCore.Mvc;

namespace CryptoTracker.Api.Controllers;

[ApiController]
[Route("sentiment")]
public class SentimentController : ControllerBase
{
    private readonly SentimentService _sentimentService;

    public SentimentController(SentimentService sentimentService)
    {
        _sentimentService = sentimentService;
    }

    [HttpGet]
    public async Task<ActionResult<List<SentimentResponse>>> GetSentiment(
        [FromQuery] SentimentQuery request)
    {
        try
        {
            List<SentimentResponse> sentiments =
                await _sentimentService.GetSentimentAsync(request);
            return Ok(sentiments);
        }
        catch (ArgumentException exception)
        {
            return BadRequest(exception.Message);
        }
    }
}
