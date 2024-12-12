using System;
using System.Collections.Generic;
using System.IO;
using System.Net.Http;
using System.Threading.Tasks;
using System.Web.Http;

namespace WebCamKeyboardAnalyzer.Controllers
{
    public class ImageAnalysisController : ApiController
    {
        private async Task<List<dynamic>> SendImageToPythonAPI(byte[] imageBytes)
        {
            using (var client = new HttpClient())
            {
                try
                {
                    var content = new MultipartFormDataContent();
                    content.Add(new ByteArrayContent(imageBytes), "image", "image.jpg");

                    // Send the image to the Python Flask API (YOLO detection)
                    var response = await client.PostAsync("http://127.0.0.1:5000/detect", content);

                    string responseContent = await response.Content.ReadAsStringAsync();

                    if (!response.IsSuccessStatusCode)
                    {
                        throw new Exception($"Error contacting Python API: {response.ReasonPhrase}, Response: {responseContent}");
                    }

                    // Deserialize the bounding box response
                    return Newtonsoft.Json.JsonConvert.DeserializeObject<List<dynamic>>(responseContent);
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error sending image to Python API: {ex.Message}");
                    throw;
                }
            }
        }

        // POST: api/ImageAnalysis
        [HttpPost]

        public async Task<IHttpActionResult> AnalyzeImage([FromBody] dynamic jsonData)
        {
            try
            {
                string image = jsonData.image;
                if (string.IsNullOrEmpty(image)) return BadRequest("Image data is missing.");

                string base64Data = image.Substring(image.IndexOf(",") + 1);
                byte[] imageBytes = Convert.FromBase64String(base64Data);

                var boundingBoxes = await SendImageToPythonAPI(imageBytes);

                if (boundingBoxes == null || boundingBoxes.Count == 0)
                {
                    return Ok(new { message = "No keyboard detected", boundingBoxes });
                }

                // Display the best match and its score in the response
                return Ok(new
                {
                    message = "Keyboard detected",
                    boundingBoxes,
                    bestMatch = boundingBoxes[0]["best_match"],
                    matchScore = boundingBoxes[0]["match_score"]
                });
            }
            catch (Exception ex)
            {
                return InternalServerError(ex);
            }
        }
    }
}