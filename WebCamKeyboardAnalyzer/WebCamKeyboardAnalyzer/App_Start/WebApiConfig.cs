using System;
using System.Collections.Generic;
using System.Linq;
using System.Web.Http;
using System.Web.Http.Cors;

namespace WebCamKeyboardAnalyzer
{
    public static class WebApiConfig
    {
        public static void Register(HttpConfiguration config)
        {
            // Enable CORS
            // var cors = new EnableCorsAttribute("*", "*", "*"); // You can specify origins, headers, and methods here
            var cors = new EnableCorsAttribute("https://localhost:44382", "*", "*");
            config.EnableCors(cors);

            // Other Web API configuration...
            config.MapHttpAttributeRoutes();
            config.Routes.MapHttpRoute(
                name: "DefaultApi",
                routeTemplate: "api/{controller}/{id}",
                defaults: new { id = RouteParameter.Optional }
            );
        }
    }
}
