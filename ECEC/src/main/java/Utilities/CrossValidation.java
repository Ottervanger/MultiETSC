package Utilities;

import java.util.ArrayList;
import java.util.Collections;

public class CrossValidation {
	
	public static ArrayList<ArrayList<Integer>> generateCV(double[] labels, int fold){
		ArrayList<Double> uniqueLabels = new ArrayList<>();
		for(double label : labels)
			if(false == uniqueLabels.contains(label))
				uniqueLabels.add(label);

		uniqueLabels.trimToSize();
		
		int[] index = new int[labels.length];
		int pos = 0;
		for(double currentLabel : uniqueLabels)
			for(int j = 0; j < labels.length; j++)
				if(labels[j] == currentLabel)
					index[pos++] = j;
		
		ArrayList<ArrayList<Integer>> cv = new ArrayList<ArrayList<Integer>>(fold);
		for(int i = 0; i < fold; i++)
			cv.add(new ArrayList<Integer>());
		
		for(int i = 0; i < index.length; i++)
			cv.get(i % fold).add(index[i]);
		
		return cv;
	}
}
