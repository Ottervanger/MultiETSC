package paper;

import java.io.FileWriter;
import java.io.Writer;

import com.opencsv.CSVWriter;

import DataStructures.DataLoader;
import DataStructures.EarlyClassifierResult;
import DataStructures.ProbabilityInformation;

public class Main_Classifier {
    
    public static void main(String[] args) {
        DataSetInformation dsInfo = new DataSetInformation();
        String dir = dsInfo.probabilityDir;
        String file = dsInfo.resultDir + "Probs_result.csv";
        try {
            Writer writer = new FileWriter(file);  
            CSVWriter csvWriter = new CSVWriter(writer); 
            
            for(int i = 0; i < dsInfo.datasetName.length; i++) {
                long startTime = System.currentTimeMillis();
                String data_dir = dir + dsInfo.datasetName[i];
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
        } catch(Exception e) {
            e.printStackTrace();
        }
    }
}
