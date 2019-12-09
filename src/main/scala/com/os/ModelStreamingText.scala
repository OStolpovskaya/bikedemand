package com.os

import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

import org.apache.spark._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.RandomForestRegressionModel
import org.apache.spark.sql.SparkSession
import org.apache.spark.streaming._
import org.apache.spark.streaming.dstream.ReceiverInputDStream

object ModelStreaming extends App {

  override def main(args: Array[String]) {

    // Creating Spark Conf and Streaming context
    val sparkConf = new SparkConf()
      .setAppName("bicyclestreaming")
      .setMaster("local[*]")

    val ssc = new StreamingContext(sparkConf, Seconds(3))

    // Creating Spark Session
    val spark = SparkSession
      .builder
      .config(sparkConf)
      .getOrCreate()

    import spark.implicits._

    // Create a DStream that will connect to hostname and port, like localhost 9999.
    // Then use nc -lk 9999 to send message:
    // 2011-01-23 19:00:00,1,0,0,1,4.92,6.06,30,19.0012
    // 2011-01-23 20:00:00,1,0,0,1,4.1,5.305,36,16.9979
    // 2011-01-23 21:00:00,1,0,0,1,4.1,5.305,36,12.998
    val textDStream: ReceiverInputDStream[String] = ssc.socketTextStream("localhost", 9999)

    // load existing model
    val rfModel = RandomForestRegressionModel.load("models/RandomForest")

    // handle each batch
    textDStream.foreachRDD(rdd => {
      if (!rdd.isEmpty()) {
        // parsing input
        val parsedRDD = rdd.flatMap(line => {
          val parts = line.split(",")
          val result = if (parts.length == 9) {

            try {

              val Array(datetime, season, holiday, workingday, weather, temp, atemp, humidity, windspeed) = parts
              val formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss")
              val localDateTime = LocalDateTime.parse(datetime, formatter)

              Some(datetime, season.toInt, holiday.toInt, workingday.toInt, weather.toInt, temp.toDouble, humidity.toInt, windspeed.toDouble, localDateTime.getHour, localDateTime.getDayOfMonth, localDateTime.getMonthValue)

            } catch {
              case e: Throwable =>
                println("Error while parsing: " + line)
                None
            }

          } else {
            println("Not enough parameters: " + line)
            None
          }
          result
        })

        // run model prediction and show results
        if (!parsedRDD.isEmpty()) {
          val data = parsedRDD.toDF("datetime", "season", "holiday", "workingday", "weather", "temp", "humidity", "windspeed", "hour", "day", "month")

          data.cache()
          println(s"INPUT DATA, size = ${data.count()}")
          data.show(30, false)

          val assembler = new VectorAssembler()
            .setInputCols(Array("season", "holiday", "workingday", "weather", "temp", "humidity", "windspeed", "hour", "day", "month"))
            .setOutputCol("features")

          val mlData = assembler.transform(data)
          val predicted = rfModel.transform(mlData)

          predicted.cache()
          println(s"PREDICTED, size = ${predicted.count()}")
          predicted.drop("features").show(30, false)
        }
      }

    })

    ssc.start() // Start the computation
    ssc.awaitTermination() // Wait for the computation to terminate
  }
}

