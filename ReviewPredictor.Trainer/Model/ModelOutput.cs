using Microsoft.ML.Data;

namespace ReviewPredictor.Model
{
    public class ModelOutput
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }

        [ColumnName("Score")]
        public float Score { get; set; }
    }
}
