package paper;

import java.io.FileWriter;
import java.io.Writer;

import com.opencsv.CSVWriter;

import DataStructures.DataLoader;
import DataStructures.EarlyClassifierResult;
import DataStructures.ProbabilityInformation;

public class Main_Classifier {
	
	public static void main(String[] args)
	{
		DataSetInformation dataset_info = new DataSetInformation();
		String dir = dataset_info.probabilityDir;
		String file = dataset_info.resultDir + "Probs_result.csv";
		try {
			Writer writer = new FileWriter(file);  
			CSVWriter csvWriter = new CSVWriter(writer); 
			
			for(int index = 0; index < dataset_info.datasetName.length; index++)
			{
				String data_dir = dir + dataset_info.datasetName[index];
				DataLoader loader = new DataLoader();
				ProbabilityInformation info = loader.loadProbabilityDataNewFormat(data_dir);
				{
					EarlyClassifier_ProbThreshold classifier = new EarlyClassifier_ProbThreshold();
					EarlyClassifierResult result = classifier.predict(info);
					writeProbResultToCSV(csvWriter, result, dataset_info.datasetName[index]);
				}
			}	
			csvWriter.close();
		}catch(Exception e){
			e.printStackTrace();
		}
	}
	
	private static void writeProbResultToCSV(CSVWriter csvWriter, 
			EarlyClassifierResult result, String name)
	{
		try {
		    int len = 9;
			String[] strs = new String[len];
			
			int index = 0;
			strs[index++] = name;
			strs[index++] = "accuracy=";
			strs[index++] = String.format("%.4f", result.accuracy);
			strs[index++] = "earliness=";
			strs[index++] = String.format("%.4f", result.earliness);
			strs[index++] = "f1_score=";
			strs[index++] = String.format("%.4f", result.f1_score);
			strs[index++] = "fcost=";
			strs[index++] = String.format("%.4f", result.fcost);
			csvWriter.writeNext(strs);
		}catch(Exception e){
			e.printStackTrace();
		}
	}
}
