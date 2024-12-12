using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Web.Http;
using System.Threading.Tasks;

namespace WebCamKeyboardAnalyzer.Controllers
{
    public class CameraController : ApiController
    {
        private readonly HttpClient _httpClient;

        public CameraController()
        {
            _httpClient = new HttpClient();
        }

        [HttpGet]
        [Route("api/camera/startstream")]
        public async Task<IHttpActionResult> StartStream()
        {
            var response = await _httpClient.GetAsync("http://localhost:5000/start-stream");
            if (response.IsSuccessStatusCode)
            {
                return Ok("Stream started");
            }
            return InternalServerError();
        }

        [HttpGet]
        [Route("api/camera/captureimage")]
        public async Task<IHttpActionResult> CaptureImage()
        {
            var response = await _httpClient.GetAsync("http://localhost:5000/capture-image");
            if (response.IsSuccessStatusCode)
            {
                var imageBytes = await response.Content.ReadAsByteArrayAsync();
                return Ok(imageBytes);  // Returning image as byte array
            }
            return InternalServerError();
        }

        [HttpGet]
        public async Task<HttpResponseMessage> GetStream()
        {
            var response = await _httpClient.GetAsync("http://localhost:5000/stream", HttpCompletionOption.ResponseHeadersRead);
            if (response.IsSuccessStatusCode)
            {
                var result = Request.CreateResponse(HttpStatusCode.OK);
                result.Content = new PushStreamContent(async (stream, content, context) =>
                {
                    await response.Content.CopyToAsync(stream);
                    await stream.FlushAsync();
                });
                result.Content.Headers.ContentType = new System.Net.Http.Headers.MediaTypeHeaderValue("multipart/x-mixed-replace; boundary=frame");
                return result;
            }
            return new HttpResponseMessage(HttpStatusCode.InternalServerError)
            {
                Content = new StringContent("Failed to connect to the camera stream.")
            };
        }


        [HttpGet]
        [Route("api/camera/stopstream")]
        public async Task<IHttpActionResult> StopStream()
        {
            var response = await _httpClient.GetAsync("http://localhost:5000/stop-stream");
            if (response.IsSuccessStatusCode)
            {
                return Ok("Stream stopped");
            }
            return InternalServerError();
        }

    }
}
