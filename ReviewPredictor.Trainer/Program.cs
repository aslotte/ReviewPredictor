using Microsoft.ML;
using MLOps.NET;
using MLOps.NET.Azure;
using ReviewPredictor.Model;
using System;
using System.Threading.Tasks;

namespace ReviewPredictor.Trainer
{
    class Program
    {
        static async Task Main(string[] args)
        {
            var connectionString = args[0];

            var mlLifeCycleManager = new MLLifeCycleManager().UseAzureStorage(connectionString);

            var experimentId = await mlLifeCycleManager.CreateExperimentAsync("Review Predictor");
            var runId = await mlLifeCycleManager.CreateRunAsync(experimentId);

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

            //Log metrics
            await mlLifeCycleManager.LogMetricAsync(runId,
                nameof(metrics.Accuracy),
                metrics.Accuracy);

            //Upload the model (artifact)

            //Check - is current model better than production model?

            //Save model as artifact for run

            Console.WriteLine("Training complete");
            Console.WriteLine($"Accuracy: {metrics.Accuracy}");
            Console.WriteLine($"F1Score: {metrics.F1Score}");

            //Save
            mlContext.Model.Save(model, trainTestSplit.TrainSet.Schema, "model.zip");
        }
    }
}
