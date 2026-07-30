using CryptoTracker.Api.Contracts.Requests;
using CryptoTracker.Api.Contracts.Responses;
using CryptoTracker.Api.Services;
using Microsoft.AspNetCore.Mvc;

namespace CryptoTracker.Api.Controllers;

[ApiController]
[Route("signals")]
public class SignalsController : ControllerBase
{
    private readonly SignalService _signalService;

    public SignalsController(SignalService signalService)
    {
        _signalService = signalService;
    }

    [HttpGet]
    public async Task<ActionResult<List<SignalResponse>>> GetSignals(
        [FromQuery] SignalQuery request)
    {
        try
        {
            List<SignalResponse> signals =
                await _signalService.GetSignalsAsync(request);
            return Ok(signals);
        }
        catch (ArgumentException exception)
        {
            return BadRequest(exception.Message);
        }
    }
}
