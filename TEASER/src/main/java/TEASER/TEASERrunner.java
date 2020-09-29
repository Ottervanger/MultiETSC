package sfa;
import sfa.classification.*;
import sfa.timeseries.TimeSeries;
import sfa.timeseries.TimeSeriesLoader;

import java.io.PrintStream;
import java.io.OutputStream;


public class TEASERrunner {
  private static String trainFile, testFile;

  public static void main(String[] args) {
    long startTime = System.currentTimeMillis();


    TEASERClassifier teaser = new TEASERClassifier();
    argParse(args, teaser);

    // Suppress undesired output
    PrintStream stdout = System.out;
    System.setOut(new PrintStream(new OutputStream() { public void write(int b){} }));

    TimeSeries.APPLY_Z_NORM = false;

    // Load data
    TimeSeries[] trainSamples = TimeSeriesLoader.loadDataset(trainFile);
    TimeSeries[] testSamples = TimeSeriesLoader.loadDataset(testFile);

    // Train and evaluate the classifier
    Classifier.Score scoreT = teaser.eval(trainSamples, testSamples);

    ParallelFor.shutdown();
    System.setOut(stdout);

    double elapsedTime = (System.currentTimeMillis() - startTime) / 1000.;
    System.out.printf(
      "Result: SUCCESS, %g, [%g, %g], 0\n",
      elapsedTime,
      scoreT.getTestEarliness(),
      1 - scoreT.getTestingAccuracy()
    );
  }

  private static void argParse(String[] args, TEASERClassifier teaser) {
    for(int i = 0; i<args.length; i++) {
      if (args[i].equals("-data")) {
        trainFile = args[++i];
        testFile = args[++i];
        continue;
      }
      teaser.setParameter(args[i].replace("-", ""), args[++i]);
    }
  }
}