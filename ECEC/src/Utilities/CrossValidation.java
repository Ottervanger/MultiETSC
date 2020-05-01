package Utilities;

import java.util.ArrayList;
import java.util.Collections;

public class CrossValidation {
	
	public static ArrayList<ArrayList<Integer>> generateCV(double[] labels, int fold){
		//ȷ���ж������
		ArrayList<Double> uniqueLabels = new ArrayList<>();
		for(int i = 0; i < labels.length; i++){
			if(false == uniqueLabels.contains(labels[i])){
				uniqueLabels.add(labels[i]);
			}
		}
		//Collections.sort(uniqueLabels);
		uniqueLabels.trimToSize();
		
		//��labels���������
		int[] index = new int[labels.length];
		int pos = 0;
		for(int i = 0; i < uniqueLabels.size(); i++){
			double currentLabel = uniqueLabels.get(i);
			for(int j = 0; j < labels.length; j++){
				if(labels[j] == currentLabel){
					index[pos++] = j;
				}
			}
		}
		
		//��fold��������
		ArrayList<ArrayList<Integer>> cv = new ArrayList<ArrayList<Integer>>(fold);
		for(int i = 0; i < fold; i++){
			cv.add(new ArrayList<Integer>());
		}
		
		for(int i = 0; i < index.length; i++){
			cv.get(i % fold).add(index[i]);
		}
		
		return cv;
	}

}
