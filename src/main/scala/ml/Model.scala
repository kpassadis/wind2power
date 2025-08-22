package ml

import zio._
import zio.stream._


case class PowerCurve(input:Seq[Double], output:Seq[Double])

case class WindPowerObservation(timestamp:String, windDirection:Double, windSpeed:Double, mw:Double, limited:Int) {
    override def toString = 
        s"$timestamp,$windDirection,$windSpeed,$mw,$limited"
}

object WindPowerObservation {
    def apply(line:String, idx:Int*):ZIO[Any, Nothing, Option[WindPowerObservation]] = 
        for {
            observation <- ZIO.attempt{
                val row = line.split(",")
                Some(WindPowerObservation(row(idx(0)), row(idx(1)).toDouble, row(idx(2)).toDouble, row(idx(3)).toDouble, row(idx(4)).toDouble.toInt))
            }.catchAll(_ => ZIO.succeed(None))
        } yield observation

    def fromFileName(filename:String):ZStream[Any, Throwable, (String, Long)] = {
        val fileStream:ZStream[Any, Throwable, Byte] = ZStream.fromFileName(filename)
        val byteToString = ZPipeline.utf8Decode >>> ZPipeline.splitLines
        val stringStream:ZStream[Any, Throwable, String] = fileStream.via(byteToString)
        val stringStreamWithIndex:ZStream[Any, Throwable, (String, Long)] = stringStream.zipWithIndex
        stringStreamWithIndex
    }

    def readAllFromFile(stringStreamWithIndex:ZStream[Any, Throwable, (String, Long)], idx:Int*):ZIO[Any, Throwable, Chunk[WindPowerObservation]] = {
        val windStream = stringStreamWithIndex
            .map{case (x,_) => x}
            .mapZIO(line => WindPowerObservation(line, idx:_*))
            .filter(_.isDefined).map(_.get)

        windStream.runCollect
    }
}

//Using lazy val is great because the result is cached once evaluated for the first time.
//But this approach does have limitations. Another alternative is to use 
//Use ZIO Ref as a caching mechanism for storing quantile values. 
//TODO: use ZIO Ref as a caching mechanism for storing the quantile values.
object RealInterval {
    def compare(interval:RealInterval, x:Double):ZIO[Any, Nothing, Int] = 
        ZIO.succeed{
            if ((x > interval.low) && (x <= interval.upper)) 0
            else if (x < interval.low) -1
            else 1
        }
}

case class RealInterval(id:Int, low:Double, upper:Double, data:Vector[WindPowerObservation]=Vector.empty) {

    def contains(x:Double):Boolean = (x > low) && (x <= upper)

    def compare(x:Double):Int = if (x < low) -1 else if (x > upper) 1 else 0

    def add(x:WindPowerObservation):RealInterval = this.copy(data = data :+ x)

    def quantile(p:Double):Option[Double] = {
        val power = data.map(_.mw)
        if (power.isEmpty) None 
        else if (power.length == 1) Some(power.head)
        else if (power.length == 2) Some(power.sum / 2.0)
        else {
            val xs = power.sorted
            val idx = math.min(math.floor(math.abs(p) * xs.length), power.length-1).toInt
            xs.length match {
                case l if l % 2 == 0 && idx < (power.length - 2) => {
                    Some((xs(idx) + xs(idx + 1)) / 2.0)
                }
                case _ => {
                    Some(xs(idx))
                }
            }
        }
    }

    lazy val median:Option[Double] = quantile(0.5)

    lazy val firstQuantile:Option[Double] = quantile(0.25)

    lazy val thirdQuantile:Option[Double] = quantile(0.75)

    lazy val iqr:Option[Double] = for {
        q3 <- quantile(0.75)
        q1 <- quantile(0.25)
    } yield q3 - q1

    def isOutlier(x: Double): Option[Int] = {
        for {
            a <- iqr
            l <- firstQuantile
            u <- thirdQuantile
        } yield {
            val lower = l - 1.5 * a
            val upper = u + 1.5 * a
            if (x < lower || x > upper) 1 else 0
        }
    }

    override def toString:String = s"[$id,$low,$upper]"
}

class SimpleKNN(val seed:Double, val nIntervals:Int, val intervals:Seq[RealInterval] = Seq.empty) {

    private def discreteStream(seed:Double, increment:Double):ZPipeline[Any, Nothing, ((Double, WindPowerObservation), Long), RealInterval] = 
        ZPipeline.map{case (_, i) => RealInterval((i+1).toInt, increment * i + seed, (i + 1) * increment, Vector.empty)}

    private def discretize(intervals:Chunk[RealInterval], minValue:Double, maxValue:Double):ZPipeline[Any, Nothing, (Double,WindPowerObservation), RealInterval] = 
        ZPipeline.map{case (x,y) => 
            val intervalOpt:Option[RealInterval] = intervals.find(interval => interval.contains(x))
            intervalOpt match {
                case Some(interval) => {
                    interval.add(y)
                }
                case None if x < minValue => {
                    intervals.head.add(y)
                } 
                case _ => intervals.last.add(y)
            }
        }


    def fit(data:Array[WindPowerObservation]):ZIO[Any, Nothing, SimpleKNN] = {
        val xStream = ZStream.fromIterable(data.map(_.windSpeed))
        val yStream = ZStream.fromIterable(data)
        val inputStream = xStream.zip(yStream)
        val finiteStream:ZStream[Any, Nothing, ((Double, WindPowerObservation), Long)] = inputStream.zipWithIndex
        for {
            minValue <- finiteStream.map(_._1._1).via(Statistics.runningMin).runCollect.map(l => l.last)
            maxValue <- finiteStream.map(_._1._1).via(Statistics.runningMax).runCollect.map(l => l.last)
            increment <- ZIO.succeed((maxValue - minValue) / nIntervals)
            intervals <- finiteStream.via(discreteStream(seed, increment)).runCollect.map(all => all.filter(interval => interval.low < maxValue))
            filledIntervals <- inputStream.via(discretize(intervals, minValue, maxValue)).runCollect.map{intervals => intervals.groupBy(_.id).map{case (id, intervals) => 
                if (intervals.length < 2){
                    intervals.head
                } else {
                    intervals.tail.foldLeft(intervals.head){case (acc, interval) => RealInterval(acc.id, acc.low, acc.upper, acc.data ++ interval.data)}
                }
            }}
            sortedIntervals = filledIntervals.toSeq.sortBy(_.low)
            model <- ZIO.succeed(new SimpleKNN(seed, nIntervals, sortedIntervals))
        } yield model
    }

    def predict(data:Array[WindPowerObservation]):ZIO[Predictor, Nothing, Seq[Option[Double]]] = for {
        x <- ZIO.succeed(data.map(_.windSpeed))
        predictor <- ZIO.service[Predictor]
        preds <- ZIO.collectAllPar(
            x.map(xi => Utils.ZFind[Double, RealInterval](xi, intervals, RealInterval.compare).flatMap(interval => predictor.predict(interval)))
        ) 
    } yield preds

    def getPowerCurve():ZIO[Predictor, Nothing, PowerCurve] = for {
        predictor <- ZIO.service[Predictor]
        input = intervals.map(interval => (interval.low + interval.upper) / 2)
        outputOpt <- ZIO.collectAll(intervals.map(interval => predictor.predict(interval)))
        output = outputOpt.map(_.getOrElse(0.0))
    } yield PowerCurve(input, output)

    override def toString:String = 
        "id,lower,upper,value,outlier\n" + intervals.map(i => i.toString).mkString("\n")
    
    def serialize(filename: String): ZIO[Any, Nothing, Unit] =  
        ZIO.acquireReleaseWith(
            ZIO.attempt {
                val file = new java.io.File(filename)
                new java.io.PrintWriter(file)
            })(writer => ZIO.succeed(writer.close()))
            { writer => ZIO.succeed(writer.println(toString))}
        .catchAll(err => Console.printLine(s"$err").orDie *> ZIO.unit)

}