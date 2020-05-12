using Microsoft.ML;
using NUnit.Framework;
using ReviewPredictor.Model;
using System;

namespace ReviewPredictor.DataTests
{
    [TestFixture]
    public class DataValidationTests
    {
        [Test]
        public void GivenValidData_ShouldReturnTrue()
        {
            //Arrange
            var mlContext = new MLContext(seed: 1);

            //Act
            IDataView dataView = mlContext.Data.LoadFromTextFile<ProductReview>(@"product_reviews.csv", hasHeader: true, separatorChar: ',');

            dataView.Preview();

            //Assert
            Assert.True(true);
        }

        [Test]
        public void GivenInValidData_ShouldThrowException()
        {
            //Arrange
            var mlContext = new MLContext(seed: 1);

            //Act
            IDataView dataView = mlContext.Data.LoadFromTextFile<ProductReview>("product_reviews - bad data.csv", hasHeader: true, separatorChar: ',');

            Action dataDelegate = () => dataView.Preview();

            //Asset
            Assert.That(dataDelegate, Throws.TypeOf<FormatException>());
        }
    }
}
