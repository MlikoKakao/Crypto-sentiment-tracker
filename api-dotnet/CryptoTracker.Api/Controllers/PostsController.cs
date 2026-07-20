using CryptoTracker.Api.Contracts.Requests;
using CryptoTracker.Api.Contracts.Responses;
using CryptoTracker.Api.Services;
using Microsoft.AspNetCore.Mvc;

namespace CryptoTracker.Api.Controllers;

[ApiController]
[Route("posts")]
public class PostsController : ControllerBase
{
    private readonly PostService _postService;

    public PostsController(PostService postService)
    {
        _postService = postService;
    }

    [HttpGet]
    public async Task<ActionResult<List<PostResponse>>> GetPosts(
        [FromQuery] PostQuery request)
    {
        try
        {
            List<PostResponse> posts = await _postService.GetPostsAsync(request);
            return Ok(posts);
        }
        catch (ArgumentException exception)
        {
            return BadRequest(exception.Message);
        }
    }
}
