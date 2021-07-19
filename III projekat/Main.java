package test3;

import java.util.HashMap;
import java.util.Map;
import com.sun.javafx.font.Metrics;
import com.univocity.parsers.csv.CsvParser;
import com.univocity.parsers.csv.CsvParserSettings;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS;
import org.apache.spark.mllib.classification.NaiveBayes;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.classification.SVMModel;
import org.apache.spark.mllib.classification.SVMWithSGD;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.regression.LinearRegressionModel;
import org.apache.spark.mllib.regression.LinearRegressionWithSGD;
import org.apache.spark.mllib.tree.DecisionTree;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;

import scala.Char;
import scala.Tuple2;
public class Main {

 public static InputStreamReader getReader(String path){
	 try{
		 return new InputStreamReader(new FileInputStream(path), "UTF-8");
	 }
	 catch(Exception e){
		 e.printStackTrace();
	 }
	 return null;
 }


 public static JavaRDD<LabeledPoint> loadData(SparkSession session, String path){

	 ArrayList<LabeledPoint> data = new ArrayList<>();
	 CsvParserSettings settings= new CsvParserSettings();
	 settings.getFormat().setLineSeparator("\n");

	 CsvParser parser = new CsvParser(settings);

	 parser.beginParsing(getReader(path));

	 String[] row;
	 while((row = parser.parseNext()) != null){
		 double label = Double.parseDouble(row[row.length-1]);
		 double[] features = new double[row.length-1];
		 for(int i = 1; i < row.length-1; i++) {
			 features[i] = row[i].compareTo("?")==0?new Double(1):Double.parseDouble(row[i].trim());
		 }
		 data.add(new LabeledPoint(label, Vectors.dense(features)));
	 }

	 JavaSparkContext jc = JavaSparkContext.fromSparkContext(session.sparkContext());

	 return jc.parallelize(data);
 }


 public static void main(String[] args) {

	 SparkSession session = SparkSession
	 .builder()
	 .appName("SparkTest")
	 .master("local")
	.getOrCreate();
	
	 SparkContext context = session.sparkContext();
	
	 JavaRDD<LabeledPoint> data = loadData(session,"data/breast-cancer-wisconsin (1).data");
	 JavaRDD<LabeledPoint>[] data1=data.randomSplit(new double[] {0.8,0.2},11L);
	 
	 JavaRDD<LabeledPoint> traningSet=data1[0].cache();
	 JavaRDD<LabeledPoint> testSet=data1[1];
	 
	 //--NaiveBayes--
	
	 NaiveBayesModel Naivemodel = NaiveBayes.train(traningSet.rdd(),1.0);
	 
	 JavaRDD<Tuple2<Object, Object>> NaivePredictions=testSet.map(p->{
		 Double pred = Naivemodel.predict(p.features());
		 return new Tuple2<>(pred,p.label());
	 });
	 MulticlassMetrics NaiveMetrics= new MulticlassMetrics(NaivePredictions.rdd());
	 BinaryClassificationMetrics NaiveBmetrics =
			  new BinaryClassificationMetrics(NaivePredictions.rdd());
	 
	 
	//--SVM--
	
	JavaRDD<LabeledPoint> newTraningSet=traningSet.map((x)->{
		if(x.label()==2) return new LabeledPoint(0,x.features());
		else return new LabeledPoint(1,x.features());		
	});
	JavaRDD<LabeledPoint> newTestSet=testSet.map((x)->{
		if(x.label()==2) return new LabeledPoint(0,x.features());
		else return new LabeledPoint(1,x.features());		
	});
	
	int numIterations = 100;
	SVMModel SVMmodel = SVMWithSGD.train(newTraningSet.rdd(), numIterations);

	//SVMmodel.clearThreshold();
	JavaPairRDD<Object, Object> SVMPredictions = newTestSet.mapToPair(p ->
	  new Tuple2<>(SVMmodel.predict(p.features()), p.label()));
	BinaryClassificationMetrics SVMBmetrics =
			  new BinaryClassificationMetrics(SVMPredictions.rdd());
	double auROC = SVMBmetrics.areaUnderROC();
	MulticlassMetrics SVMMetrics =new MulticlassMetrics(SVMPredictions.rdd());
	 
	
	//--LogisticRegression--
	LogisticRegressionModel LRmodel = new LogisticRegressionWithLBFGS()
			  .setNumClasses(2)
			  .run(newTraningSet.rdd());
	JavaPairRDD<Object, Object> LRpredictions = newTestSet.mapToPair(p ->
	  new Tuple2<>(LRmodel.predict(p.features()), p.label()));
	MulticlassMetrics LRmetrics = new MulticlassMetrics(LRpredictions.rdd());
	BinaryClassificationMetrics LRBmetrics =
			  new BinaryClassificationMetrics(LRpredictions.rdd());


	//--Decision tree

	int numClasses = 2;
	Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<>();
	String impurity = "gini";
	int maxDepth = 5;
	int maxBins = 32;

	DecisionTreeModel DTmodel = DecisionTree.trainClassifier(newTraningSet, numClasses,
			  categoricalFeaturesInfo, impurity, maxDepth, maxBins);
	
	JavaPairRDD<Object, Object> DTpredictions = newTestSet.mapToPair(p ->
	  new Tuple2<>(DTmodel.predict(p.features()), p.label()));
	double testErr =
			DTpredictions.filter(pl -> !pl._1().equals(pl._2())).count() / (double) newTestSet.count();

	
	MulticlassMetrics DTmetrics = new MulticlassMetrics(DTpredictions.rdd());
	BinaryClassificationMetrics DTBmetrics =
			  new BinaryClassificationMetrics(DTpredictions.rdd());
//	 
//	 JavaRDD<Tuple2<Object, Object>> precision = bmetrics.precisionByThreshold().toJavaRDD();
//	 System.out.println("Precision by threshold: " + precision.collect());
//
//
//	 JavaRDD<?> recall = bmetrics.recallByThreshold().toJavaRDD();
//	 System.out.println("Recall by threshold: " + recall.collect());
//	 
//	 
//
//	// F Score by threshold
//	JavaRDD<?> f1Score = bmetrics.fMeasureByThreshold().toJavaRDD();
//	System.out.println("F1 Score by threshold: " + f1Score.collect());
//
//	JavaRDD<?> f2Score = bmetrics.fMeasureByThreshold(2.0).toJavaRDD();
//	System.out.println("F2 Score by threshold: " + f2Score.collect());
//
//	// Precision-recall curve
//	JavaRDD<?> prc = bmetrics.pr().toJavaRDD();
//	System.out.println("Precision-recall curve: " + prc.collect());
//
//	// Thresholds
//	JavaRDD<Double> thresholds = precision.map(t -> Double.parseDouble(t._1().toString()));

	// ROC Curve



	 try {
		 FileWriter fw = new FileWriter("statistics1.txt");
	
		//--Naive Bayes
		 fw.write("Naive Bayes \n\nAccuracy = " + NaiveMetrics.accuracy()+ "\n");

		 Matrix confusion = NaiveMetrics.confusionMatrix();
		 fw.write("Confusion matrix: \n" + confusion+"\n");

		 for(int i=0; i<NaiveMetrics.labels().length; i++)
		 {
			 fw.write("Class"+ NaiveMetrics.labels()[i]+ "  precision ="+NaiveMetrics.precision(NaiveMetrics.labels()[i])+ "\n");
			 fw.write("Class"+ NaiveMetrics.labels()[i]+ "  recal ="+NaiveMetrics.recall(NaiveMetrics.labels()[i])+ "\n");
			 fw.write("Class"+ NaiveMetrics.labels()[i]+ "  F score ="+NaiveMetrics.fMeasure(NaiveMetrics.labels()[i])+ "\n");

		 }

		JavaRDD<?> roc = NaiveBmetrics.roc().toJavaRDD();
		fw.write("ROC curve: " + roc.collect()+"\n");
		 
	 
		//--Logistic Regression

		 fw.write("Logistic regression \n\nAccuracy = " + LRmetrics.accuracy()+ "\n");

		 Matrix LRconfusion = LRmetrics.confusionMatrix();
		 fw.write("Confusion matrix: \n" + LRconfusion+"\n");
		 
		 for(int i=0; i<LRmetrics.labels().length; i++)
		 {
			 fw.write("Class"+ LRmetrics.labels()[i]+ "  precision ="+LRmetrics.precision(LRmetrics.labels()[i])+ "\n");
			 fw.write("Class"+ LRmetrics.labels()[i]+ "  recal ="+LRmetrics.recall(LRmetrics.labels()[i])+ "\n");
			 fw.write("Class"+ LRmetrics.labels()[i]+ "  F score ="+LRmetrics.fMeasure(LRmetrics.labels()[i])+ "\n");

		 }
		 
		 //--Decision Tree
		 fw.write("DecisionTree \n\nAccuracy = " + DTmetrics.accuracy()+ "\n");
		 fw.write("Test error" + testErr+"\n");
		 Matrix DTconfusion = DTmetrics.confusionMatrix();
		 fw.write("Confusion matrix: \n" + DTconfusion+"\n");
		 
		 for(int i=0; i<DTmetrics.labels().length; i++)
		 {
			 fw.write("Class"+ DTmetrics.labels()[i]+ "  precision ="+DTmetrics.precision(DTmetrics.labels()[i])+ "\n");
			 fw.write("Class"+ DTmetrics.labels()[i]+ "  recal ="+DTmetrics.recall(DTmetrics.labels()[i])+ "\n");
			 fw.write("Class"+ DTmetrics.labels()[i]+ "  F score ="+DTmetrics.fMeasure(DTmetrics.labels()[i])+ "\n");

		 }
		
		 //--SVM
		 fw.write("SVM \n\nAccuracy = " + SVMMetrics.accuracy()+ "\n");

		 Matrix SVMconfusion = SVMMetrics.confusionMatrix();
		 fw.write("Confusion matrix: \n" + SVMconfusion+"\n");
		 
		 for(int i=0; i<SVMMetrics.labels().length; i++)
		 {
			 fw.write("Class"+ SVMMetrics.labels()[i]+ "  precision ="+SVMMetrics.precision(SVMMetrics.labels()[i])+ "\n");
			 fw.write("Class"+ SVMMetrics.labels()[i]+ "  recal ="+SVMMetrics.recall(SVMMetrics.labels()[i])+ "\n");
			 fw.write("Class"+ SVMMetrics.labels()[i]+ "  F score ="+SVMMetrics.fMeasure(SVMMetrics.labels()[i])+ "\n");

		 }
		
		
	 	fw.close();
	
	 } catch (IOException ex) {
		 Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
	 }
	
	 
//--SVM
//	 System.out.println("SVM \n\nAccuracy = " + SVMMetrics.accuracy()+ "\n");
//
//	 Matrix SVMconfusion = SVMMetrics.confusionMatrix();
//	 System.out.println("Confusion matrix: \n" + SVMconfusion);
//	 
//	 for(int i=0; i<NaiveMetrics.labels().length; i++)
//	 {
//		 System.out.format("Class %.0f precision = %f\n", SVMMetrics.labels()[i],SVMMetrics.precision(SVMMetrics.labels()[i]));
//		 System.out.format("Class %.0f recal = %f\n", SVMMetrics.labels()[i],SVMMetrics.recall(SVMMetrics.labels()[i]));
//		 System.out.format("Class %.0f F score = %f\n", SVMMetrics.labels()[i],SVMMetrics.fMeasure(SVMMetrics.labels()[i]));
//
//	 }
//	 

	 
//	JavaRDD<Tuple2<Object, Object>> SVMprecision = SVMBmetrics.precisionByThreshold().toJavaRDD();
//	System.out.println("Precision by threshold: " + SVMprecision.collect());
//	
//	System.out.println("Area under ROC = " + auROC);
//	
	 //primer pamcenja modela
		System.out.println("testerr"+testErr);

	 Naivemodel.save(context, "C:/spark-models/NaiveModel");
	 session.close();

 }

 private static void writePredictions(JavaRDD<Double> preds) {
	 try {
		 FileWriter fw = new FileWriter("predictions.csv");
	
		 List<Double> predsDouble = preds.takeOrdered((int)preds.count());
	
	 for(Double d : predsDouble){
		 fw.write(d.toString() + "\n");
	 }
	 	fw.close();
	
	 } catch (IOException ex) {
		 Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
	 }
	 }
}