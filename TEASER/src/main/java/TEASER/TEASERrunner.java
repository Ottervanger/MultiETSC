package sfa;
import sfa.classification.*;
import sfa.timeseries.TimeSeries;
import sfa.timeseries.TimeSeriesLoader;

import java.util.*;

public class TEASERrunner {
  private static String trainFile, testFile;

  public static void main(String[] args) {
    long startTime = System.currentTimeMillis();
    argParse(args);

    TimeSeries.APPLY_Z_NORM = false;

    TimeSeries[] trainSamples = TimeSeriesLoader.loadDataset(trainFile);
    TimeSeries[] testSamples = TimeSeriesLoader.loadDataset(testFile);

    TEASERClassifier t = new TEASERClassifier();

    Classifier.Score scoreT = t.eval(trainSamples, testSamples);
    ParallelFor.shutdown();

    double elapsedTime = (System.currentTimeMillis() - startTime) / 1000.;
    System.out.printf(
      "Result: SUCCESS, %g, [%g, %g], 0\n",
      elapsedTime,
      scoreT.getTestEarliness(),
      scoreT.getTestingAccuracy()
    );
  }

  private static void argParse(String[] args) {
    for(int i = 0; i<args.length; i++) {
      if (args[i].equals("-data")) {
        trainFile = args[++i];
        testFile = args[++i];
        continue;
      }
    }
  }
}