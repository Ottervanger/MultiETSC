package DataStructures;

import java.util.ArrayList;
import java.util.Collections;

import java.io.*;

public class ProbabilityInformation implements Serializable {
	
	public int[] trainLength;
	public int[] testLength;
	
	public int[][] trainStepLength;
	public int[][] testStepLength;
	
	public double[][][] trainProbs;
	public double[][][] testProbs;
	
	public double[] trainLabels;
	public double[] testLabels;
	
	public double[] labelTypes;
	
	public int trainNum;
	public int testNum;
	public int stepNum;
	public int labelNum;

	public static ProbabilityInformation fromFile(String filename) throws IOException {
		try {
			FileInputStream f = new FileInputStream(new File(filename));
			ObjectInputStream o = new ObjectInputStream(f);
			ProbabilityInformation pri = (ProbabilityInformation) o.readObject();
			o.close();
			f.close();
			return pri;
		} catch (FileNotFoundException e) {
			System.out.println("File not found");
		} catch (IOException e) {
			System.out.println("Error initializing stream");
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
		}
		throw new IOException("Failed to load instance");
	}

	public void toFile(String filename) {
		try {
			FileOutputStream f = new FileOutputStream(new File(filename));
			ObjectOutputStream o = new ObjectOutputStream(f);
			o.writeObject(this);
			o.close();
			f.close();
		} catch (FileNotFoundException e) {
			System.out.println("File not found");
		} catch (IOException e) {
			System.out.println("Error initializing stream");
		}
	}
	
	public void postprocess() {
		ArrayList<Double> labels = new ArrayList<>();
		for(int i = 0; i < trainLabels.length; i++){
			double label = trainLabels[i];
			if(labels.indexOf(label) < 0){
				labels.add(label);
			}
		}
		Collections.sort(labels);

		labelTypes = new double[labels.size()];
		for(int i = 0; i < labels.size(); i++){
			labelTypes[i] = labels.get(i).doubleValue();
		}
		
		trainNum = trainLabels.length;
		testNum = testLabels.length;
		stepNum = trainProbs[0].length;
		labelNum = labelTypes.length;
	}
	
	public double[] labels() {
		return labelTypes;
	}
}
