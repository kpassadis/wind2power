//> using dep "dev.zio::zio::2.1.20"
//> using dep "dev.zio::zio-streams::2.1.20"
//> using dep "org.scalanlp:breeze_2.13:2.1.0"
//> using dep "org.apache.commons:commons-math3:3.6.1"
//> using dep "io.github.pityka::nspl-awt:0.11.1"
//> using dep "com.stripe::rainier-core:0.3.5"
//> using scala "2.13"

//scala-cli run Main.sc src/main/scala
import org.nspl._ 
import org.nspl.awtrenderer._ 
import ml._ 
import zio._
import zio.stream._
import Utils._


