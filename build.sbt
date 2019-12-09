name := "bikedemand"

version := "0.1"

scalaVersion := "2.11.12"

libraryDependencies += "log4j" % "log4j" % "1.2.14"
libraryDependencies += "org.scalatest" %% "scalatest" % "3.1.0-RC3" % Test
libraryDependencies += "org.apache.spark" %% "spark-core" % "2.4.4"
libraryDependencies += "org.apache.spark" %% "spark-sql" % "2.4.4"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.4.4"
