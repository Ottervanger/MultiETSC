package DataStructures;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import com.opencsv.CSVReader;

public class DataLoader {
	
	public TimeSeriesSet loadSourceData(String dataset) {
	    ArrayList<TimeSeriesInstance> alldata = new ArrayList<>();

	    int index = 0;
	    File file = new File(dataset);
	    try (BufferedReader br = new BufferedReader(new FileReader(file))) {
	    	
	    	String line = null;
	    	while ((line = br.readLine()) != null) {
	    		if (line.startsWith("@")) {
	    			continue;
	    		}

	    		//separator may be " " or "," in dataset
	    		String separator = line.contains(",") ? "," : " ";
	    		String[] columns = line.split(separator);
	    		int columnLength = columns.length;
	    		//some instance may be filled in with "NaN"
	    		for(int i = 0; i < columns.length; i++){
	    			String value = columns[i].trim();
	    			if(value.equals("NaN")){
	    				columnLength = i + 1;
	    				break;
	    			}
	    		}

	    		double[] data = new double[columnLength - 1];
	    		//first is label
	    		double label = Double.parseDouble(columns[0].trim());
	    		//next is data
	    		int i = 1;
	    		for(; i < columnLength; i++){
	    			data[i-1] = Double.parseDouble(columns[i].trim());
	    		}

	    		if (i > 1) {
	    			TimeSeriesInstance ts = new TimeSeriesInstance(
	    					Arrays.copyOfRange(data, 0, i-1), label, index++, i);
	    			//ts.normalize();
	    			alldata.add(ts);
	    		}
	    	}

	    } catch (IOException e) {
	    	e.printStackTrace();
	    }

	    TimeSeriesSet tsSet = new TimeSeriesSet(alldata.toArray(new TimeSeriesInstance[]{}));
	    System.out.println("Reading from " + dataset);
	    System.out.println("Number of class is: " + tsSet.labelset.length);
	    System.out.println("Number of instances is: " + alldata.size());
	    System.out.println("Length max: " + tsSet.getMaxLength() + ", min: " + tsSet.getMinLength() + ", average: " + tsSet.getAverageLength());
	    System.out.println("=====================================================");
	    return tsSet;
	}

	public ProbabilityInformation loadProbabilityDataNewFormat(String dataset) {
		ProbabilityInformation infor = new ProbabilityInformation();
		int step_num = 20;
		int probs_index = 4;
		
		try {
			//train
			for(int i = 1; i <= step_num; i++) {
				String train_file = dataset + File.separator + "general-train-probs-" + Integer.toString(i) + ".csv";
				Reader reader = new FileReader(train_file);  
			    CSVReader csvReader = new CSVReader(reader);
			    List<String[]> all_data = csvReader.readAll();
			    csvReader.close();
			    
			    if (i == 1) {
			    	infor.trainProbs = new double[all_data.size()][step_num][all_data.get(0).length - probs_index];
			    	infor.trainLabels = new double[all_data.size()];
			    	infor.trainLength = new int[all_data.size()];	    	
			    	infor.trainStepLength = new int[all_data.size()][step_num];
			    }
			    
			    for(int index = 0; index < all_data.size(); index++) {
			    	infor.trainLabels[index] = Double.valueOf(all_data.get(index)[1]);
			    	infor.trainLength[index] = Integer.valueOf(all_data.get(index)[2]);
			    	infor.trainStepLength[index][i-1] = Integer.valueOf(all_data.get(index)[3]);
			    	for(int c = 0; c < all_data.get(0).length - probs_index; c++) {
			    		infor.trainProbs[index][i-1][c] = Double.valueOf(all_data.get(index)[c+probs_index]);
			    	}
			    }
			}
			
			//test
			for(int i = 1; i <= step_num; i++) {
				String test_file = dataset + File.separator + "general-test-probs-" + Integer.toString(i) + ".csv";
				Reader reader = new FileReader(test_file);  
			    CSVReader csvReader = new CSVReader(reader);
			    List<String[]> all_data = csvReader.readAll();
			    csvReader.close();
			    
			    if (i == 1) {
			    	infor.testProbs = new double[all_data.size()][step_num][all_data.get(0).length - probs_index];
			    	infor.testLabels = new double[all_data.size()];
			    	infor.testLength = new int[all_data.size()];
			    	infor.testStepLength = new int[all_data.size()][step_num];
			    }
			    
			    for(int index = 0; index < all_data.size(); index++) {
			    	infor.testLabels[index] = Double.valueOf(all_data.get(index)[1]);
			    	infor.testLength[index] = Integer.valueOf(all_data.get(index)[2]);
			    	infor.testStepLength[index][i-1] = Integer.valueOf(all_data.get(index)[3]);
			    	for(int c = 0; c < all_data.get(0).length - probs_index; c++) {
			    		infor.testProbs[index][i-1][c] = Double.valueOf(all_data.get(index)[c+probs_index]);
			    	}
			    }
			}
			
			infor.postprocess();
		}catch(Exception e){
			e.printStackTrace();
		}
		return infor;
	}
}
