package com.Harmonycloud.NWClassifier
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{MultilayerPerceptronClassificationModel, MultilayerPerceptronClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, Word2Vec}
import org.apache.spark.sql.{Column, Row, SQLContext, SparkSession}
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import java.sql.{Connection, Date, DriverManager, PreparedStatement, ResultSet, Timestamp}
import java.text.SimpleDateFormat
import java.util.Calendar

import Array._
import com.hankcs.hanlp.tokenizer.NLPTokenizer
import org.apache.http.client.methods.HttpOptions
import org.apache.spark.{SparkConf, SparkContext}

object AbClassifier {
  final val VECTOR_SIZE = 200
  //mysql读写

  //get a connection to mysql database
  def getConnection():Connection={
    /*val driver = "com.mysql.jdbc.Driver"
    Class.forName(driver)*/
    DriverManager.getConnection("jdbc:mysql://10.10.101.115:3306/ai_ops","root","123456")
  }


  //分词：全角转半角、停用词处理、分词、存储
  def segment(sc:SparkContext): Unit = {
    //stop words
    val stopWordPath = "停用词路径"
    val bcStopWords = sc.broadcast(sc.textFile(stopWordPath).collect().toSet)

    //content segment
    val inPath = "训练语料路径"
    val segmentRes = sc.textFile(inPath)
      .map(AsciiUtil.sbc2dbcCase) //全角转半角
      .mapPartitions(it =>{
      it.map(ct => {
        try {
          val nlpList = NLPTokenizer.segment(ct)
          import scala.collection.JavaConverters._
          nlpList.asScala.map(term => term.word)
            .filter(!bcStopWords.value.contains(_))
            .mkString(" ")
        } catch {
          case e: NullPointerException => println(ct);""
        }
      })
    })

    //save segment result
    segmentRes.saveAsTextFile("分词结果路径")
    bcStopWords.unpersist()
  }




  def main(args: Array[String]) {
    /*if (args.length < 4) {
      println("Usage:master mode(train) File-Path")
      println("Usage:master mode(test) File-Path ID")
      sys.exit(1)
    }*/
    //LogUtils.setDefaultLogLevel()



    def myFun(iterator: Iterator[(Row)]): Unit = {
      //insert abnormal_out into database
      var conn: Connection = null
      val today = new java.util.Date()
      val sql_date = new Timestamp(today.getTime)
      var ps: PreparedStatement = null
      val sql = "insert into t_job_out_profile(job_id,job_log_line,time_stamp,feedback_status) values (?, ?, ?, ?);"
      try {
        conn = DriverManager.getConnection("jdbc:mysql://10.10.101.115:3306/ai_ops?useUnicode=true&characterEncoding=UTF-8", "root", "123456")
        iterator.foreach(Row => {
          ps = conn.prepareStatement(sql)
          ps.setInt(1, args(3).toInt)
          println( Row.toString().getClass.getSimpleName )
          ps.setString(2, Row.toString())
          ps.setTimestamp(3, sql_date)
          ps.setInt(4, 0)

          ps.executeUpdate()
        }
        )
      } catch {
        case e: Exception => println(e)
      } finally {
        if (ps != null) {
          ps.close()
        }
        if (conn != null) {
          conn.close()
        }
      }
    }





    val conf = new SparkConf().setMaster("local").setAppName("SMS Message Classification (HAM or SPAM)")

    //val conf = new SparkConf().setAppName("SMS Message Classification (HAM or SPAM)")
    val sc = new SparkContext(conf)
    //sc.addJar("/root/mysql-connector-java-6.0.4.jar")
    val sqlCtx = new SQLContext(sc)

    if(args(1)=="train") {
      if (args.length < 4) {
        println("Usage:master mode(train) File-Path ID")
        sys.exit(1)
      }
        train()

      }
    else if(args(1)=="run"){
      if (args.length < 4) {
        println("Usage:master mode(run) File-Path ID")
        sys.exit(1)
      }
        test()
    }
    else{
      println("Usage:master mode(train/run) File-Path")
      println("Usage:master mode(run) File-Path ID")
      sys.exit(1)
    }


    //test function
    def test():Unit = {
      println("Test function")

      val schema = StructType(
        Array(
          StructField("message",StringType,true)
        )
      )

      val parsedRDD = sc.textFile(args(2)).map(eachRow => {
        ("ham",eachRow.split(" "))
      })
      val msgDF = sqlCtx.createDataFrame(parsedRDD).toDF("label","message")

      msgDF.printSchema
      msgDF.select("label","message").show(30)

      //将单列test.txt文件转换成Dataframe
      val labelIndexer = new StringIndexer()
        .setInputCol("label")
        .setOutputCol("indexedLabel")
        .fit(msgDF)

      val labelConverter = new IndexToString()
        .setInputCol("prediction")
        .setOutputCol("predictedLabel")
        .setLabels(labelIndexer.labels)


      //val layers = Array[Int](VECTOR_SIZE,50,20,10,2)

      val model=PipelineModel.load("hdfs://10.10.101.115:9000/mllib/multi-NW/model/MultiNW_model.model")//加载模型地址

      val predictionResultDF = model.transform(msgDF)

      predictionResultDF.show(30)

      var resultDF=predictionResultDF.filter("predictedLabel = 'spam'").select("message").toDF("message")

      val sparkSession = SparkSession.builder.getOrCreate()
      import sparkSession.implicits._
      var transformMessDF=resultDF.as[(Array[String])]
       .map { case (message) => (message.mkString(", ") ) }
       .toDF( "message")
      transformMessDF.printSchema
      transformMessDF.select("message").show(30)
      /*
      val time1=System.currentTimeMillis()
      时间戳
      transformMessDF.repartition(1).write.csv("data/test"+time1+".csv")//异常输出保存地址
      */
      //var Job_out_Path="hdfs://10.10.101.115:9000/mllib/multi-NW/output/test"+args(3).toString+".csv"
      //println(Job_out_Path)
      //transformMessDF.repartition(1).write.csv(Job_out_Path)//异常输出保存地址
      /*val connection_test = getConnection()//invoke a function to get a connection

      var run_sql="insert into t_job_out_profile values(" + args(3).toInt+",'" + Job_out_Path + "');"
      println(run_sql)
      val prepareSta_test: PreparedStatement = connection_test.prepareStatement(run_sql);
      prepareSta_test.executeUpdate()
      */
      transformMessDF.rdd.foreachPartition(myFun)//插入数据库

    }

    //train function
    def train(): Unit = {

      val parsedRDD = sc.textFile(args(2)).map(_.split("\t")).map(eachRow => {
        (eachRow(0),eachRow(1).split(" "))//对message中的sentence进行按空格分词

      })
      println(parsedRDD)
      val msgDF = sqlCtx.createDataFrame(parsedRDD).toDF("label","message")

      msgDF.printSchema
      msgDF.select("label","message").show(30)

      val labelIndexer = new StringIndexer()
        .setInputCol("label")
        .setOutputCol("indexedLabel")
        .fit(msgDF)

      val word2Vec = new Word2Vec()
        .setInputCol("message")
        .setOutputCol("features")
        .setVectorSize(VECTOR_SIZE)
        .setMinCount(1)

      val layers = Array[Int](VECTOR_SIZE,80,5,2)
      val mlpc = new MultilayerPerceptronClassifier()
        .setLayers(layers)
        .setBlockSize(512)
        .setSeed(1234L)
        .setMaxIter(512)
        .setFeaturesCol("features")
        .setLabelCol("indexedLabel")
        .setPredictionCol("prediction")

      val labelConverter = new IndexToString()
        .setInputCol("prediction")
        .setOutputCol("predictedLabel")
        .setLabels(labelIndexer.labels)

      val Array(trainingData, testData) = msgDF.randomSplit(Array(0.9, 0.1))

      val pipeline = new Pipeline().setStages(Array(labelIndexer,word2Vec,mlpc,labelConverter))
      val model = pipeline.fit(trainingData)
      val time1=System.currentTimeMillis()//获取时间戳

      //var Model_save_Path="hdfs://10.10.101.115:9000/mllib/multi-NW/model/MultiNW_"+time1+"_"+args(3).toString+"model.model"
      model.write.overwrite().save("hdfs://10.10.101.115:9000/mllib/multi-NW/model/MultiNW_model.model")//保存模型地址

      val predictionResultDF = model.transform(testData)
      //below 2 lines are for debug use
      predictionResultDF.printSchema
      predictionResultDF.select("message","label","predictedLabel").show(30)

      val evaluator = new MulticlassClassificationEvaluator()
        .setLabelCol("indexedLabel")
        .setPredictionCol("prediction")
        //  .setMetricName("precision")
        //Since SPARK-15617 deprecated precision in MulticlassClassificationEvaluator, many ML examples broken.
        //We should use accuracy to replace precision in these examples.
        .setMetricName("accuracy")
      val predictionAccuracy = evaluator.evaluate(predictionResultDF)
      println("Testing Accuracy is %2.4f".format(predictionAccuracy * 100) + "%")


      //the data should be inserted into MySQL
      //the database is named "mydatabase",the table's name is employee
      //the database and table should be created in advance
      val connection = getConnection()//invoke a function to get a connection
      val prepareSta: PreparedStatement = connection.prepareStatement("insert into t_model_profile values("+args(3).toInt+"," + predictionAccuracy + ",'0');");
      /*val preparedStatement: PreparedStatement = connection.prepareStatement(
        "select * from t_train_job" )
      val result: ResultSet = preparedStatement.executeQuery()
      println("id\tname\tstatus")
      while(result.next()){
        print(result.getString("id")+" ")
        print(result.getString("name")+" ")
        print(result.getString("status")+" ")
        println()
      }
      */
      prepareSta.executeUpdate()

    }



  }
}
