package ml

import zio._
import zio.stream._

object Utils {

    def unsafeRun[E,A](zio:ZIO[Any, E,A]):A = Unsafe.unsafe{implicit unsafe => Runtime.default.unsafe.run(zio).getOrThrowFiberFailure()}

    def writeCsv(header:String, lines:Seq[String], filename:String):ZIO[Any, Throwable, Unit] = {
        val stringStream:ZStream[Any, Nothing, String] =ZStream.fromIterable(header +: lines)
        val result:ZIO[Any, Throwable, Unit] = stringStream
            .intersperse("\n")                     
            .via(ZPipeline.utf8Encode)             
            .run(ZSink.fromFile(new java.io.File(filename))) 
            .unit

        result
    }

    def sinkWithScores(inFile:String, outFile:String, model:Weibull, idx:Int*):ZIO[Any, Throwable, Unit] = {
        val fileStream:ZStream[Any, Throwable, Byte] = ZStream.fromFileName(inFile)
        val byteToString = ZPipeline.utf8Decode >>> ZPipeline.splitLines
        val stringStream:ZStream[Any, Throwable, String] = fileStream.via(byteToString)
        val stringStreamWithIndex:ZStream[Any, Throwable, (String, Long)] = stringStream.zipWithIndex
        val observationStream:ZStream[Any, Throwable, (Option[WindPowerObservation], String, Long)] = 
            stringStreamWithIndex
                .mapZIO{case (line, i) => WindPowerObservation(line, idx:_*).flatMap(o => ZIO.succeed((o, line, i)))}
        val cleanObservation:ZStream[Any, Throwable, (WindPowerObservation, String, Long)] = 
            observationStream
                .filter(t => t._1.isDefined)
                .map(t => (t._1.get, t._2, t._3))
                .filter(t => t._1.limited == 0)

        val stringStreamOut:ZStream[Any, Throwable, String] = cleanObservation.mapZIOPar(2){case (obs, line, i) => 
            val score = Statistics.anomaly_score(obs.windSpeed, obs.mw, model.beta,model.eta)
            //val absoluteScore = math.abs(score)
            ZIO.succeed(s"$i,${obs.toString},$score") 
        }
        //ZSink.fromFile returns a ZSink[Any, Throwable, Byte, Byte, Long], which means:
        //It writes Bytes to the file It returns the number of bytes written as a Long
        //But the code expects a sink that returns Unit, so there's a type mismatch.
        //If you don't care about how many bytes were written and just want ZIO[Any, Throwable, Unit], 
        //you can discard the result with .unit
        val result:ZIO[Any, Throwable, Unit] = stringStreamOut
            .intersperse("\n")                     
            .via(ZPipeline.utf8Encode)             
            .run(ZSink.fromFile(new java.io.File(outFile))) 
            .unit

        result
    }

    object ZFind {
        def apply[A,B](value:A, list:Seq[B], compare:(B,A) => ZIO[Any, Nothing, Int]):ZIO[Any, Nothing, B] = {
            def go(list:Seq[B], x:A):ZIO[Any, Nothing, B] = ZIO.ifZIO(ZIO.succeed(list.length == 1))(
                onTrue = ZIO.succeed(list.head),
                onFalse = for {
                    midItem <- ZIO.succeed{
                        val idx:Int = list.length / 2
                        list(idx)
                    }
                    item <- ZIO.ifZIO(compare(midItem, value).map(res => res == 0))(
                        onTrue = ZIO.succeed(midItem),
                        onFalse = ZIO.ifZIO(compare(midItem, value).map(res => res > 0))(
                            onFalse = go(list.dropRight(list.length / 2), value),
                            onTrue = go(list.drop(list.length / 2), value)
                        )
                    )
                    
                } yield item
            )

            go(list, value)
        }
    }


    def unfold[A,B](seed:A)(stop:A=>Boolean, next:A=>A, f:A=>B)(seq:Seq[B]):Seq[B] = {
        if (stop(seed)) seq else
        unfold(next(seed))(stop, next, f)(seq :+ f(seed))
    }

}