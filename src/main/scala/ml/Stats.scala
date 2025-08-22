package ml

import zio._
import zio.stream._
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunction
import org.apache.commons.math3.analysis.MultivariateFunction
import org.apache.commons.math3.optim._
import org.apache.commons.math3.optim.nonlinear.scalar._
import org.apache.commons.math3.optim.nonlinear.scalar.noderiv._

object Statistics {
    //The cool thing about this implementation of the running min is that I can also determine the position where the minimum was located.
    def runningMin:ZPipeline[Any, Nothing, Double, Double] = ZPipeline.mapAccum(Double.MaxValue){case (currentMin, x) => 
        if (x < currentMin) {
            (x, x)
        } else {
            (currentMin, currentMin)
        }
    }

    //Similarly I can define a pipeline that calculates the maximum value
    def runningMax:ZPipeline[Any, Nothing, Double, Double] = ZPipeline.mapAccum(Double.MinValue){case (currentMax, x) => 
        if (x > currentMax) {
            (x, x)
        } else {
            (currentMax, currentMax)
        }
    }

    def systematicSample[A](inputStream:ZStream[Any, Nothing, A], n:Int):ZStream[Any, Nothing, A] = {
        inputStream
            .zipWithIndex
            .filter{case (_, i) => i % n == 0}
            .map(t => t._1)
    }

    def anomaly_score(v: Double, p: Double, beta: Double, eta: Double) : Double = {
        val dist1 = new Weibull(eta, beta)
        val cdf_v = dist1.cdf(v)
        val eta1 = math.pow(eta, 3.0)
        val beta1 = beta / 3.0
        val dist2 = new Weibull(eta1, beta1)
        val cdf_p = dist2.cdf(p)
        math.log(+ 1e-8 + (cdf_v / (cdf_p + 1e-8)))
    }

}

trait Sampler[A] {
    val inputStream:ZStream[Any, Nothing, A]
    def sample:ZStream[Any, Nothing, A]
}

class TrainTestSplit[A, B](override val inputStream: ZStream[Any, Nothing, A],p: Double)(implicit ev: A <:< (B, Int)) extends Sampler[A] {

  override def sample: ZStream[Any, Nothing, A] =
    inputStream.mapZIO { a =>
      val x = ev(a) 
      for {
        isTrain <- Random.nextDoubleBetween(0.0, 1.0).map(_ < p)
        data <- ZIO.ifZIO(ZIO.succeed(isTrain))(
          onTrue  = ZIO.succeed(a),
          onFalse = ZIO.succeed((x._1, x._2 + 1).asInstanceOf[A])
        )
      } yield data
    }
}

class Weibull(val eta:Double, val beta:Double) {

  def fit(data:Vector[Double]):Weibull = {

    def weibullNegLogLikelihood(eta: Double, beta: Double): Double = {
      if (eta <= 0 || beta <= 0) 1e10
      else {
        val n = data.length
        val logSum = data.map(math.log).sum
        val expSum = data.map(x => math.pow(x / eta, beta)).sum
        -(n * math.log(beta) - n * beta * math.log(eta) + (beta - 1.0) * logSum - expSum)
      }
    }
    
    val objective = new ObjectiveFunction(new MultivariateFunction {
    override def value(point: Array[Double]): Double =
      weibullNegLogLikelihood(point(0), point(1))
    })

    val initialGuess = new InitialGuess(Array(eta, beta))

    val simplex = new NelderMeadSimplex(2)
    val optimizer = new SimplexOptimizer(1e-8, 1e-8)

    val result = optimizer.optimize(
      objective,
      GoalType.MINIMIZE,
      initialGuess,
      simplex,
      new MaxEval(1000)
    )

    val est = result.getPoint
    new Weibull(est(0), est(1))

  }

  def cdf(x: Double): Double = {
    if (x <= 0.0) 0.0
    else 1.0 - math.exp(-math.pow(x / eta, beta))
  }
  
}