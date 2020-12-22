package ECEC;

import sfa.classification.ParallelFor;
import sfa.timeseries.TimeSeries;
import sfa.timeseries.TimeSeriesLoader;

import java.io.PrintStream;
import java.io.OutputStream;

public class ECECRunner {
  private static String trainFile, testFile;
  
  public static void main(String[] args) {
    long startTime = System.currentTimeMillis();


    ECECClassifier ecec = new ECECClassifier();
    argParse(args, ecec);

    // Suppress undesired output
    PrintStream stdout = System.out;
    System.setOut(new PrintStream(new OutputStream() { public void write(int b){} }));

    TimeSeries.APPLY_Z_NORM = false;

    // Load data
    TimeSeries[] trainSamples = TimeSeriesLoader.loadDataset(trainFile);
    TimeSeries[] testSamples = TimeSeriesLoader.loadDataset(testFile);

    // Train and evaluate the classifier
    ECECClassifier.Result result = ecec.fitAndTest(trainSamples, testSamples);

    ParallelFor.shutdown();
    System.setOut(stdout);

    double elapsedTime = (System.currentTimeMillis() - startTime) / 1000.;
    System.out.printf(
      "Result: SUCCESS, %g, [%g, %g], 0\n",
      elapsedTime,
      result.earliness,
      1 - result.accuracy
    );
  }

  private static void argParse(String[] args, ECECClassifier ecec) {
    for(int i = 0; i<args.length; i++) {
      if (args[i].equals("-data")) {
        trainFile = args[++i];
        testFile = args[++i];
        continue;
      }
      ecec.setParameter(args[i].replace("-", ""), args[++i]);
    }
  }
}
