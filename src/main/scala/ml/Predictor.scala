package ml

import zio._

trait Predictor {
    def predict(interval:RealInterval):ZIO[Any, Nothing, Option[Double]]
}

case class QuantilePredictor(p:Double) extends Predictor {
    override def predict(interval:RealInterval):ZIO[Any, Nothing, Option[Double]] = ZIO.succeed(interval.quantile(p))
} 

object QuantilePredictor {
    def live(p:Double):ZLayer[Any, Nothing, Predictor] = 
        ZLayer.succeed(QuantilePredictor(p))
}