package paper;

import java.io.File;
import java.io.IOException;

import Classifiers.sfa.classification.ParallelFor;
import Classifiers.sfa.timeseries.TimeSeries;
import Classifiers.sfa.timeseries.TimeSeriesLoader;

import DataStructures.ProbabilityInformation;

public class Main_TrainProbability {
    
    public static void main(String[] args){
        long startTime = System.currentTimeMillis();
        String dir = "/scratch/ottervanger/UCR/";
        String dataset = "CBF";

        ProbabilityInformation probInfo;
        try {
            probInfo = ProbabilityInformation.fromFile("out/" + dataset);
        } catch (IOException e) {
            String trainFile = dir + dataset + File.separator + dataset + "_TRAIN.tsv";
            String testFile = dir + dataset + File.separator + dataset + "_TEST.tsv";
            
            TimeSeries.APPLY_Z_NORM = false;
            TimeSeries[] trainSamples = TimeSeriesLoader.loadDataset(trainFile);
            TimeSeries[] testSamples = TimeSeriesLoader.loadDataset(testFile);
            ProbabilityTrainer_General_Memory classifer = new ProbabilityTrainer_General_Memory();
            probInfo = classifer.process(trainSamples, testSamples);
            probInfo.toFile("out/" + dataset);
        }

        EarlyClassifier_ProbThreshold classifier2 = new EarlyClassifier_ProbThreshold();
        EarlyClassifier_ProbThreshold.Result result = classifier2.predict(probInfo);

        double elapsedTime = (System.currentTimeMillis() - startTime) / 1000.;
        System.out.printf(
          "Result: SUCCESS, %g, [%g, %g], 0\n",
          elapsedTime,
          result.earliness,
          1 - result.accuracy
        );

        ParallelFor.shutdown();
    }
}
