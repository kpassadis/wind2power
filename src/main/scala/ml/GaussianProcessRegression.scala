package ml

import breeze.linalg._
import breeze.numerics._
import java.awt.geom.Ellipse2D

class GaussianProcessRegression private(val parameters:Array[Array[Double]], val measurements:Array[Double], val sigma:Double, val y_mean:Double, val y_std:Double) {

    def kernel(v1:Array[Double], v2:Array[Double]):Double = {
        val distance_squared:Double = v1.zip(v2).map(t => (t._1 - t._2) * (t._1 - t._2)).sum
        math.exp(-distance_squared / (2 * sigma * sigma))
    }

    //A numerically stable version. 
    def estimate(query:Array[Double]):(Double, Double) = {
        //A vector: calculate the kernel value for each parameter (similar to what we did to calculate the weights in the previous version)
        val kernelVector:Array[Double] = parameters.map(p => kernel(p, query)).toArray
        val kernelVectorB:DenseVector[Double] = DenseVector(kernelVector:_*)
        //A square matrix: it calculates the kernel between all pairs of existing parameters
        val kernelMatrix:Array[Array[Double]] = parameters.map(p1 => parameters.map(p2 => kernel(p1, p2)))
        val kernelMatrixB:DenseMatrix[Double] = DenseMatrix(kernelMatrix:_*)
        //Add a very small regularization to the kernel matrix to ensure numerical stability in the 
        //attempt to calculate the inverse. 
        val stabilizedMatrix = kernelMatrixB + DenseMatrix.eye[Double](parameters.length) * 1e-6
        val weights: DenseVector[Double] = (kernelVectorB.t * inv(stabilizedMatrix)).t
        val mu = weights.toArray.zip(measurements).map{case (w, y) => w*y}.sum + y_mean
        val variance = kernel(query, query) - weights.dot(kernelVectorB)
        val uncertainty = math.sqrt(math.max(variance, 1e-10)) // to avoid sqrt of negative
        //Return the expectation and the uncertainty. As a sanity check
        //run the predict method on the parameters and it should return 
        //numbers very close to measurements with uncertainty close to zero.
        (mu, uncertainty)
    }

    def predict(sample:Array[Array[Double]]):Array[(Double, Double)] = 
        sample.map(query => estimate(query)).toArray

    def predict():Array[Double]= measurements

}

object GaussianProcessRegression{
    def apply(parameters:Array[Array[Double]], measurements:Array[Double], sigma:Double):GaussianProcessRegression = {
        val N = measurements.length
        val y_mean = measurements.sum / N
        val y_std = measurements.map(y => math.sqrt((y - y_mean) * (y - y_mean) / (N-1)) ).sum
        val y = measurements.map(y => y - y_mean)
        
        new GaussianProcessRegression(parameters, y, sigma, y_mean, y_std)
    }
}