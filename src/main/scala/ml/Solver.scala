package ml

import zio._

trait Solver[A, B, R] {
    val problem:Problem[A]
    def step(candidate:A, epoch:Int):ZIO[R,Nothing, A]
    def optimize(epochs:Int):ZIO[R, Nothing, Seq[B]]
}

trait GenericSolver[A, B] extends Solver[A, B, Any] 
trait BayesianSolver[A, B, R <: Acquisition] extends Solver[A, B, R]

class BayesianOptimization(val problem:Problem[Array[Double]], nIter:Int, parameters:Ref[List[Array[Double]]], measurements:Ref[Array[Double]], sigma:Double) extends BayesianSolver[Array[Double], (Array[Double], Double), Acquisition] {
    
    //The step function implements the so called inner loop of the Bayesian optimization. In the inner loop we seek to optimize the acquisition function.
    //At the end of the inner loop we return the values of the parameters that optimize the acquisition function.
    override def step(candidate:Array[Double], epoch:Int):ZIO[Acquisition, Nothing, Array[Double]] = for {
        acqObj <- ZIO.service[Acquisition]
        parameterArray <- ZIO.ifZIO(parameters.get.map(_.isEmpty))(
            onTrue = problem.generate(),
            onFalse = for {
                parameterArray <- parameters.get
                measurementArray <- measurements.get
                gpr = GaussianProcessRegression(parameterArray.toArray, measurementArray, sigma)   
                acqValue = acqObj.objective(candidate, gpr)
                ref <- Ref.make[Array[Double]](candidate)
                _ <- ZIO.foreachPar((0 until nIter).toArray){i => 
                    for {   
                        newCandidate <- problem.generate(candidate)
                        newAqcValue = acqObj.objective(newCandidate, gpr)
                        _ <- ref.update{p => 
                            if (newAqcValue < acqValue) {
                                newCandidate
                             } else p
                        }
                    } yield ()
                }
                optimalParam <- ref.get
            } yield optimalParam
        )
        
    } yield parameterArray
    
    //This is the outer loop of the optimization process. We execute the inner loop, where the acquisition function is optimized.
    override def optimize(epochs:Int):ZIO[Acquisition, Nothing, Seq[(Array[Double], Double)]] = for {
        _ <- ZIO.foreach((0 until epochs).toArray){epoch => 
            for {
                candidate <- problem.generate()
                _ <- Console.printLine(s"Generated random candidate ${candidate.mkString(",")}").orDie
                proposal <- step(candidate, epoch)
                _ <- Console.printLine(s"Inner loop solution ${proposal.mkString(",")}").orDie
                cost <- problem.evaluate(proposal)
                _ <- measurements.update(arr => arr :+ cost)
                _ <- parameters.update(arr => arr :+ proposal)
            } yield ()
        }
        chain <- parameters.get
        evaluations <- measurements.get
    } yield (chain zip evaluations)
}