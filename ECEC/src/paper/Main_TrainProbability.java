package paper;

import java.io.File;

import DataStructures.DataLoader;
import DataStructures.TimeSeriesInstance;
import DataStructures.TimeSeriesSet;
import Utilities.ParallelFor;

public class Main_TrainProbability {
	
	public static void main(String[] args){
		DataSetInformation dataset_info = new DataSetInformation();
		String dir = dataset_info.sourceDir;  
		String result_dir = dataset_info.probabilityDir;
		for(int index = 0; index < dataset_info.datasetName.length; index++){
			String dataset = dataset_info.datasetName[index];
            String train_file = dir + dataset + File.separator + dataset + "_TRAIN";
            String test_file = dir + dataset + File.separator + dataset + "_TEST";
            
            String result_dir2 = result_dir + dataset;
            File file = new File(result_dir2);  
            if(!file.exists()){  
                file.mkdirs();  
            }
            
            DataLoader loader = new DataLoader();
            TimeSeriesSet train_data = loader.loadSourceData(train_file);
            TimeSeriesSet test_data = loader.loadSourceData(test_file);
            
            ProbabilityTrainer_General_Memory classifer = new ProbabilityTrainer_General_Memory();
            classifer.process(train_data, test_data, result_dir2);    
		}
		ParallelFor.shutdown();
	}
}
