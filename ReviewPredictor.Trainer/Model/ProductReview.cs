using Microsoft.ML.Data;

namespace ReviewPredictor.Model
{
    public class ProductReview
    {
        [LoadColumn(0)]
        public bool Sentiment;

        [LoadColumn(1)]
        public string Review;
    }
}
