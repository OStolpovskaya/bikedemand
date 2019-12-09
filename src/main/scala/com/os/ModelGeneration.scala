package com.os

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.{GBTRegressor, LinearRegression, RandomForestRegressor}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

/**
  * Created by Olga Stolpovskaia OXS2KTT on 2019-11-05
  */
object ModelGeneration extends App {

  val sparkSession = SparkSession.builder.master("local[*]").getOrCreate()
  val sc = sparkSession.sparkContext

  import sparkSession.implicits._

  val raw = sparkSession
    .read
    .option("header", "true")
    .option("timestampFormat", "dd-MM-yyyy HH:mm") //01-01-2011 00:00
    .option("inferSchema", "true")
    .csv("files/train.csv")
    .cache()

  println("-=== Data analysis ===-")

  println(s"Data frame: raw data, size = ${raw.count()}")
  raw.show(10, false)

  raw.printSchema()

  println("Grouped by season")
  raw.groupBy('season).agg(floor(mean('count)).as("mean")).sort('season).show()

  println("Grouped by weather")
  raw.groupBy('weather).agg(floor(mean('count)).as("mean")).sort('weather).show()

  println("Grouped by hour")
  raw.groupBy(hour('datetime).as("hour")).agg(floor(mean('count)).as("mean")).sort('hour).show()

  println("Grouped by month")
  raw.groupBy(month('datetime).as("month")).agg(floor(mean('count)).as("mean")).sort('month).show()

  println("Grouped by holiday")
  raw.groupBy('holiday).agg(floor(mean('count)).as("mean")).sort('holiday).show()

  println("Grouped by working day")
  raw.groupBy('workingday).agg(floor(mean('count)).as("mean")).sort('workingday).show()

  println("-=== Data wrangling ===-")

  val data = raw
    .withColumn("hour", hour('datetime))
    .withColumn("day", dayofmonth('datetime))
    .withColumn("month", month('datetime))
    .withColumn("label", 'count)
    .drop("datetime", "atemp", "casual", "registered", "count") // remove correlated columns

  data.cache()
  println(s"Data frame: data, size = ${data.count()}")
  data.show(10, false)

  val columns = Array("season", "holiday", "workingday", "weather", "temp", "humidity", "windspeed", "hour", "day", "month")
  //"season", "holiday", "workingday", "weather", "temp", "humidity", "hour", "day", "month"

  val assembler = new VectorAssembler()
    .setInputCols(columns)
    .setOutputCol("features")

  val mlData = assembler.transform(data)

  val df = mlData.select((columns ++ Array("features", "label")).map(n => col(n)): _*).cache()
  println(s"Data frame: data ready for ml, size = ${df.count()}")
  df.show(10, false)

  val Array(trainData, testData) = df.randomSplit(Array(0.8, 0.2))
  val evaluator = new RegressionEvaluator().setMetricName("rmse")

 println("-=== LINEAR REGERESSION ===-")

  val lr = new LinearRegression()
    .setMaxIter(100)

  val lrModel = lr.fit(trainData)
  val lrPredictions = lrModel.transform(testData)

  lrPredictions.cache()
  println(s"Data frame: lrPredictions, size = ${lrPredictions.count()}")
  lrPredictions.show(10, false)

  var lrRMSE: Double = evaluator.evaluate(lrPredictions)
  println(s"Root Mean Squared Error (RMSE) on test data = ${lrRMSE}")

  println("-=== RANDOM FOREST ===-")
  val rf = new RandomForestRegressor()

  val rfModel = rf.fit(trainData)
  val rfPredictions = rfModel.transform(testData)

  rfPredictions.cache()
  println(s"Data frame: rfPredictions, size = ${rfPredictions.count()}")
  rfPredictions.show(10, false)

  val rfRMSE = evaluator.evaluate(rfPredictions)
  println(s"Root Mean Squared Error (RMSE) on test data = $rfRMSE")

  println("-=== GRADIENT BOOSTED REGR ===-")
  // Train a GBT model.
  val gbt = new GBTRegressor()
    .setMaxIter(10)

  val gbtmodel = gbt.fit(trainData)
  val gbtpredictions = gbtmodel.transform(testData)

  gbtpredictions.cache()
  println(s"Data frame: gbtPredictions, size = ${gbtpredictions.count()}")
  gbtpredictions.show(10, false)

  val gbtRSME = evaluator.evaluate(gbtpredictions)
  println(s"Root Mean Squared Error (RMSE) on test data = $gbtRSME")

println("-=== Logistic regression ===-")
  val logr = new LogisticRegression()
    .setMaxIter(10)
    .setRegParam(0.3)
    .setElasticNetParam(0.8)

  // Fit the model
  val logrModel = logr.fit(trainData)
  val logrPredictions = logrModel.transform(testData)

  logrPredictions.cache()
  println(s"Data frame: logrPredictions, size = ${logrPredictions.count()}")
  logrPredictions.show(10, false)

  val logrRSME = evaluator.evaluate(logrPredictions)
  println(s"Root Mean Squared Error (RMSE) on test data = $logrRSME")

  println("-=== Comparing results ===-")

  println(s"Linear regression: $lrRMSE")
  println(s"Random forest: $rfRMSE")
  println(s"Gradient boosted regr: $gbtRSME")
  println(s"Logistic regression: $logrRSME")

  println("-=== Saving model of Random Forest ===-")
  rfModel.write.overwrite().save("models/RandomForest")


}
