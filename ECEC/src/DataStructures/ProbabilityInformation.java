package DataStructures;

import java.util.ArrayList;
import java.util.Collections;

public class ProbabilityInformation{

	public ProbabilityInformation(){
		
	}
	
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
