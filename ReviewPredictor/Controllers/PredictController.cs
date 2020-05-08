using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.ML;
using ReviewPredictor.Model;

// For more information on enabling MVC for empty projects, visit https://go.microsoft.com/fwlink/?LinkID=397860

namespace ReviewPredictor
{
    [Route("[controller]")]
    public class PredictController : Controller
    {
        private readonly PredictionEnginePool<ProductReview, ModelOutput> predictionEnginePool;

        public PredictController(PredictionEnginePool<ProductReview, ModelOutput> predictionEnginePool)
        {
            this.predictionEnginePool = predictionEnginePool;
        }

        // GET: /<controller>/
        public IActionResult Index()
        {
            return View();
        }


        [HttpPost]
        public bool Predict(ProductReview productReview)
        {
            return this.predictionEnginePool.Predict(productReview).Prediction;
        }
    }
}
