package paper;

import DataStructures.DataLoader;
import DataStructures.EarlyClassifierResult;
import DataStructures.ProbabilityInformation;

public class Main_Classifier {
    
    public static void main(String[] args) {
        long startTime = System.currentTimeMillis();

        String dir = "./out/probability/";
        String dataset = "ECG200";

        String data_dir = dir + dataset;
        DataLoader loader = new DataLoader();
        ProbabilityInformation info = loader.loadProbabilityDataNewFormat(data_dir);

        EarlyClassifier_ProbThreshold classifier = new EarlyClassifier_ProbThreshold();
        EarlyClassifierResult result = classifier.predict(info);

        double elapsedTime = (System.currentTimeMillis() - startTime) / 1000.;
        System.out.printf(
          "Result: SUCCESS, %g, [%g, %g], 0\n",
          elapsedTime,
          result.earliness,
          1 - result.accuracy
        );
    }
}
