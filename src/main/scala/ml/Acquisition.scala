package ml

import breeze.linalg._
import breeze.numerics._
import breeze.stats.distributions._  
import org.apache.commons.math3.random.MersenneTwister     
import scala.util.Failure
import scala.util.Success
import scala.util.Try
import scala.util.Random
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunction
import org.apache.commons.math3.analysis.MultivariateFunction
import org.apache.commons.math3.optim._
import org.apache.commons.math3.optim.nonlinear.scalar._
import org.apache.commons.math3.optim.nonlinear.scalar.noderiv._
import zio._


object Functions {

    implicit val randBasis: RandBasis = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(42)))

    //A univariate function with multiple local minima that we want to minimize.
    def f(x:Array[Array[Double]], noise:Double):Array[Double] = {
        val g = new Gaussian(0,1)
        val sample = g.sample(x.length)
        val xflat = x.flatMap(_.toList)
        xflat.zip(sample).map{ case (xi, r) =>
            -math.sin(3 * xi) - math.pow(xi, 2.0) + 0.2 * xi + noise * r
        }
    }
    
}

trait Acquisition {
    //val gpr:GaussianProcessRegression 
    val bounds: Array[(Double, Double)]
    val xi:Double
    def objective(x:Array[Double], gpr:GaussianProcessRegression):Double
    def fit(initialGuess:Array[Double], gpr:GaussianProcessRegression):Array[Double] = {
        implicit val randBasis: RandBasis = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(42)))
         //Define the objective function as a minimization problem.
        val objectiveFunc = new ObjectiveFunction(new MultivariateFunction{
            override def value(point: Array[Double]): Double = -objective(point, gpr)
        })

        val lower = bounds.map(_._1)
        val upper = bounds.map(_._2) 
        val dim = bounds.length
        val popSize = new CMAESOptimizer.PopulationSize(4 + (3 * math.log(dim)).toInt)
        val sigmas  = new CMAESOptimizer.Sigma(bounds.map{ case (lo,hi) => 0.2 * (hi - lo) })

        val opt = new CMAESOptimizer(
            1000, 1e-9, true, 0, 0, new org.apache.commons.math3.random.MersenneTwister(42),
            false, new SimpleValueChecker(1e-8, 1e-8)
        )

        val res = opt.optimize(
            new MaxEval(10000),
            objectiveFunc,
            GoalType.MINIMIZE,
            new InitialGuess(initialGuess),
            new SimpleBounds(lower, upper),
            popSize, sigmas
        )

        res.getPoint
    }
}

//Define the typeclass
trait AcquisitionFactory[R <: Acquisition] {
  def create(bounds: Array[(Double, Double)], xi: Double): R
}

object Acquisition {

  //Defin etypeclass instances
  implicit val eiFactory: AcquisitionFactory[ExpectedImprovement] = new AcquisitionFactory[ExpectedImprovement] {
    def create(bounds: Array[(Double, Double)], xi: Double): ExpectedImprovement = new ExpectedImprovement(bounds, xi)
  }

  implicit val lcbFactory: AcquisitionFactory[LowerConfidenceBound] = new AcquisitionFactory[LowerConfidenceBound] {
    def create(bounds: Array[(Double, Double)], xi: Double): LowerConfidenceBound = new LowerConfidenceBound(bounds, xi)
  }  

  def live[R <: Acquisition : Tag](bounds: Array[(Double, Double)], xi: Double)(implicit factory: AcquisitionFactory[R]): ZLayer[Any, Nothing, R] = {
    ZLayer.succeed(factory.create(bounds, xi))
  }
}


class ExpectedImprovement(override val bounds: Array[(Double, Double)], override val xi:Double=0.0) extends Acquisition {
    override def objective(x:Array[Double], gpr:GaussianProcessRegression ):Double = {
        implicit val randBasis: RandBasis = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(42)))
        val g = new Gaussian(0,1)
        val prediction = gpr.estimate(x)
        val mu = prediction._1
        val sigma = prediction._2
        val estimates = gpr.predict()
        val muOpt = estimates.max
        val imp = mu -  muOpt - xi
        val z = Try(imp / sigma).toOption.getOrElse(0.0)
        val ei = if (sigma < 1e-8) 0.0 else imp * g.cdf(z) + sigma * g.pdf(z)
        ei
    }
}

class LowerConfidenceBound(override val bounds: Array[(Double, Double)], override val xi:Double=0.0) extends Acquisition {
    override def objective(x:Array[Double], gpr:GaussianProcessRegression):Double = {
        val (y, sigmaY) = gpr.estimate(x)
        val lcb = y - xi * sigmaY
        lcb
    }
} 