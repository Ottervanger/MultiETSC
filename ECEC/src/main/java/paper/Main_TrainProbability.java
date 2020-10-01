package paper;

import java.io.File;

import DataStructures.DataLoader;
import DataStructures.TimeSeriesInstance;
import DataStructures.TimeSeriesSet;
import Utilities.ParallelFor;

public class Main_TrainProbability {
    
    public static void main(String[] args){
        String dir = "/scratch/ottervanger/UCR/";
        String result_dir = "./out/probability/";

        String dataset = "ECG200";
        String train_file = dir + dataset + File.separator + dataset + "_TRAIN.tsv";
        String test_file = dir + dataset + File.separator + dataset + "_TEST.tsv";
        
        String result_dir2 = result_dir + dataset;
        File file = new File(result_dir2);  
        if (!file.exists())
            file.mkdirs();
        
        TimeSeriesSet train_data = DataLoader.loadSourceData(train_file);
        TimeSeriesSet test_data = DataLoader.loadSourceData(test_file);
        
        ProbabilityTrainer_General_Memory classifer = new ProbabilityTrainer_General_Memory();
        classifer.process(train_data, test_data, result_dir2);

        ParallelFor.shutdown();
    }
}
