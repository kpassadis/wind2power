package ml

import zio._

trait Problem[A] {
    def evaluate(candidate:A):UIO[Double]
    def generate():UIO[A]
    def generate(candidate:A):UIO[A] = ZIO.succeed(candidate)
}


class QuantileOptimizationProblem(k:Int, trainData:Array[WindPowerObservation], testData:Array[WindPowerObservation]) extends Problem[Array[Double]] {
    
    val capacity = 50

    override def evaluate(candidate:Array[Double]):UIO[Double] = {
        val q = candidate(0)
        val predictor = QuantilePredictor.live(q)
        (for {
        _ <- Console.printLine("Fitting model").orDie
        model <- new SimpleKNN(0.0, k).fit(trainData)
        _ <- Console.printLine(s"Predicting on test set with size ${testData.length}").orDie
        predictions <- model.predict(testData)
        tuples = testData.map(_.mw).zip(predictions)
        cost <- ZIO.succeed{
            val (totalCost, n) = tuples.foldLeft((0.0, 0)){
                case ((acc, n), (yi, Some(yhat))) => (acc + math.abs((yi - yhat) / capacity), n+1)
                case ((acc, n), _) => (acc, n) 
            }
            100 * (totalCost / n)
        }
        _ <- Console.printLine(s"Done predicting, calculating NMAPE:$cost for model parameters: $k and $q").orDie
    } yield cost).provideLayer(predictor)
    }
    
    override def generate():UIO[Array[Double]] = ZIO.collectAll(Array.fill(1)(Random.nextDoubleBetween(0.5, 0.7)))
    
    override def generate(candidate:Array[Double]):UIO[Array[Double]] = {
        val stepSize = 0.001
        ZIO.foreach(candidate) {case c =>
            for {
                r <- Random.nextDoubleBetween(0.0, 1.0)
            } yield c + stepSize * r
        }
    }
}

object QuantileOptimizationProblem {

    def live(k:Int):ZLayer[TrainTestSplit[(WindPowerObservation, Int), WindPowerObservation], Nothing, QuantileOptimizationProblem] =
        ZLayer.fromZIO {
            for {
                sampler <- ZIO.service[TrainTestSplit[(WindPowerObservation, Int), WindPowerObservation]]
                samples <- sampler.sample.runCollect
                data = samples.toList
                (trainDataTuple, testDataTuple) = data.partition(t => t._2 == 0)
                trainData = trainDataTuple.map(_._1).toArray
                testData = testDataTuple.map(_._1).toArray
            } yield new QuantileOptimizationProblem(50, trainData, testData)
        }
}
