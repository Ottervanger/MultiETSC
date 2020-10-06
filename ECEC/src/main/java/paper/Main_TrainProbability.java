package paper;

import java.io.File;

import DataStructures.DataLoader;
import DataStructures.TimeSeriesInstance;
import DataStructures.TimeSeriesSet;
import Classifiers.sfa.classification.ParallelFor;
import DataStructures.EarlyClassifierResult;
import DataStructures.ProbabilityInformation;

public class Main_TrainProbability {
    
    public static void main(String[] args){
        long startTime = System.currentTimeMillis();
        String dir = "/scratch/ottervanger/UCR/";
        String result_dir = "./out/probability/";

        String dataset = "CBF";
        String train_file = dir + dataset + File.separator + dataset + "_TRAIN.tsv";
        String test_file = dir + dataset + File.separator + dataset + "_TEST.tsv";
        
        String result_dir2 = result_dir + dataset;
        File file = new File(result_dir2);  
        if (!file.exists())
            file.mkdirs();
        
        TimeSeriesSet train_data = DataLoader.loadSourceData(train_file);
        TimeSeriesSet test_data = DataLoader.loadSourceData(test_file);
        
        ProbabilityTrainer_General_Memory classifer = new ProbabilityTrainer_General_Memory();
        EarlyClassifier_ProbThreshold classifier2 = new EarlyClassifier_ProbThreshold();
        EarlyClassifierResult result = classifier2.predict(classifer.process(train_data, test_data, result_dir2));

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
