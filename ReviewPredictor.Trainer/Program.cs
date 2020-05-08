using Microsoft.ML;
using ReviewPredictor.Model;
using System;

namespace ReviewPredictor.Trainer
{
    class Program
    {
        static void Main(string[] args)
        {
            var mlContext = new MLContext(seed: 1);

            //Load the data
            IDataView dataView = mlContext.Data.LoadFromTextFile<ProductReview>("product_reviews.csv", hasHeader: true, separatorChar: ',');

            var trainTestSplit = mlContext.Data.TrainTestSplit(dataView);

            //Transform
            var dataProcessPipeline = mlContext.Transforms.Text.FeaturizeText("Review")
                .Append(mlContext.Transforms.CopyColumns("Features", "Review"))
                .Append(mlContext.Transforms.NormalizeMinMax("Features", "Features"));

            //Train
            var trainer = mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression(labelColumnName: "Sentiment", featureColumnName: "Features");

            var trainingPipeline = dataProcessPipeline.Append(trainer);

            Console.WriteLine("Starting training");

            ITransformer model = trainingPipeline.Fit(trainTestSplit.TrainSet);

            //Evaluate
            var predicitions = model.Transform(trainTestSplit.TestSet);
            var metrics = mlContext.BinaryClassification.Evaluate(predicitions, labelColumnName: "Sentiment");

            Console.WriteLine("Training complete");
            Console.WriteLine($"Accuracy: {metrics.Accuracy}");
            Console.WriteLine($"F1Score: {metrics.F1Score}");

            //Save
            mlContext.Model.Save(model, trainTestSplit.TrainSet.Schema, "model.zip");
        }
    }
}
