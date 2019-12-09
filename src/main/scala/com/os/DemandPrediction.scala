package com.os

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.RandomForestRegressionModel
import org.apache.spark.sql.functions.{dayofmonth, hour, month}
import org.apache.spark.sql.{SaveMode, SparkSession}

/**
  * Created by Olga Stolpovskaia OXS2KTT on 2019-11-12
  */
object DemandPrediction extends App{
  val sparkSession = SparkSession.builder.master("local[*]").getOrCreate()
  val sc = sparkSession.sparkContext

  import sparkSession.implicits._

  val raw = sparkSession
    .read
    .option("header", "true")
    .option("timestampFormat", "yyyy-MM-dd HH:mm:ss") // 2011-01-20 00:00:00 differs from train data!!!
    .option("inferSchema", "true")
    .csv("files/test.csv")
    .cache()

  println("-=== Data analysis ===-")

  println(s"Data frame: raw data, size = ${raw.count()}")
  raw.show(10, false)

  raw.printSchema()

  println("-=== Data wrangling ===-")

  val data = raw
    .withColumn("hour", hour('datetime))
    .withColumn("day", dayofmonth('datetime))
    .withColumn("month", month('datetime))

  data.cache()
  println(s"Data frame: data, size = ${data.count()}")
  data.show(10, false)

  val columns = Array("season", "holiday", "workingday", "weather", "temp", "humidity", "windspeed", "hour", "day", "month")

  val assembler = new VectorAssembler()
    .setInputCols(columns)
    .setOutputCol("features")

  val mlData = assembler.transform(data)

  println("-=== Loading model and make predictions ===-")

  val rfModel = RandomForestRegressionModel.load("models/RandomForest")
  val rfPredictions = rfModel.transform(mlData)

  rfPredictions.cache()
  println(s"Data frame: rfPredictions, size = ${rfPredictions.count()}")
  rfPredictions.show(10, false)

  println("-=== Save predictions ===-")
  rfPredictions
    .select('datetime, 'prediction.as("count"))
    .write.mode(SaveMode.Overwrite)
    .option("header", "true")
    .option("timestampFormat", "yyyy-MM-dd HH:mm:ss")
    .csv("result")

}
