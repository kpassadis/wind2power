package ml


import com.stripe.rainier.core._
import com.stripe.rainier.compute._
import com.stripe.rainier.sampler._

object BSTS {
    def apply(y:List[Double], n:Int, warmup:Int=500, chainLength:Int=1000):Seq[Double] = {
        val n = y.length
        val uErr = Normal(0,1).latentVec(n)
        val vErr = Normal(0,1).latentVec(n)
        val sLevel = LogNormal(0,1).latent
        val sSlope = LogNormal(0,1).latent
        val v0 = Normal(0, 10).latent
        val u0 = Normal(0, 1).latent
        val sObs = LogNormal(0,1).latent
        
        val v = (1 until y.length).foldLeft(List(v0)){case (acc, i) => 
            val vCurr = acc(i-1) + sSlope * vErr(i)
            acc :+ vCurr
        }

        val u = (1 until y.length).foldLeft(List(u0)){case (acc, i) => 
            val uCurr = acc(i-1) + v(i-1) + sLevel * uErr(i)
            acc :+ uCurr
        }

        val model = Model.observe(y, Vec.from(u).map{ui => Normal(ui, sObs)})
        val sampler = EHMC(warmup, chainLength)

        val trace = model.sample(sampler)

        val generator = Generator(Normal(u(n-1), sObs))
        trace.predict(generator).take(n)
    }
}


class AR2(val phi1:Real, val phi2:Real, val sigmaY:Real, val X_train:Seq[Double]) {
    
    def fit():AR2 = {
        val x = X_train
        val T = x.length

        val mu = (2 until T).map{t => 
            phi1 * x(t-1) + phi2 * x(t-2)
        }

        val mu_vec = Vec.from(mu)

        val model = Model.observe(x.drop(2), mu_vec.map(mu => Normal(mu, sigmaY)))
        val maximum_likelihood_estimators = model.optimize(List(phi1, phi2, sigmaY))
        val phi1_ml = maximum_likelihood_estimators(0)
        val phi2_ml = maximum_likelihood_estimators(1)
        val sigmaY_ml = maximum_likelihood_estimators(2)

        new AR2(phi1_ml, phi2_ml, sigmaY_ml, X_train)

    }

    def apply(n:Int):Seq[Double] = {
        
        val T = X_train.length
        val sampleSize = 30
        val mu = phi1 * X_train(T-1) + phi2 * X_train(T-2)
        val pred = Model.sample(Normal(mu, sigmaY).latent).take(sampleSize).sum / sampleSize
        
        val res:Seq[Double] = Utils.unfold((0, X_train, pred))(
            t => t._1 == n, 
            t => {
                val i = t._1
                val X_train = t._2.dropRight(1) :+ t._3
                val mu = phi1 * X_train(T-1) + phi2 * X_train(T-2)
                val pred = Model.sample(Normal(mu, sigmaY).latent).take(sampleSize).sum / sampleSize    
                (i+1, X_train, pred) 
            },
            t => t._3
        )(Seq.empty[Double])
       res
    }

}

object AR2 {
    def apply(X_train:Seq[Double]):AR2 = {
        val sigmaY = Gamma(10, 10).latent
        val phi1 = Normal(0.0, 1.0).latent
        val phi2 = Normal(0.0, 1.0).latent
        new AR2(phi1, phi2, sigmaY, X_train)
    }
}

