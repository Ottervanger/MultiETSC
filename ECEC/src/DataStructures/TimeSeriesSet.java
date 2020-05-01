package DataStructures;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

import Classifiers.sfa.timeseries.TimeSeries;
import de.bwaldvogel.liblinear.FeatureNode;
import de.bwaldvogel.liblinear.Problem;

public class TimeSeriesSet {

	public TimeSeriesInstance[] dataset = null;
	
	public double[] labelset = null;
	
	private int minLength = 0;
	private int maxLength = 0;
	private int averageLength = 0;
	
	public TimeSeriesSet(TimeSeriesInstance[] dataset){
		this.dataset = dataset;
		ArrayList<Double> labels = new ArrayList<>();
		for(int i = 0; i < dataset.length; i++){
			double label = dataset[i].label;
			if(labels.indexOf(label) < 0){
				labels.add(label);
			}
		}
		Collections.sort(labels);
		//labelset = labels.toArray(new double[labels.size()]);
		labelset = new double[labels.size()];
		for(int i = 0; i < labels.size(); i++){
			labelset[i] = labels.get(i).doubleValue();
		}
		
		minLength = minLength();
		maxLength = maxLength();
		averageLength = averageLength();
	}
	
	public TimeSeriesInstance get(int index) {
		return dataset[index];
	}
	
	public TimeSeriesSet subset(ArrayList<Integer> index){
		TimeSeriesInstance[] data = new TimeSeriesInstance[index.size()];
		for(int i = 0; i < index.size(); i++){
			data[i] = this.dataset[index.get(i)].clone();
		}
		return new TimeSeriesSet(data);
	}
	
	public TimeSeriesSet truncateTo(int length){
		TimeSeriesInstance[] data = new TimeSeriesInstance[dataset.length];
		for(int i = 0; i < dataset.length; i++){
			data[i] = dataset[i].truncateTo(length);
		}
		return new TimeSeriesSet(data);
	}
	
	public TimeSeriesSet truncateTo(double ratio, int minLength){
		TimeSeriesInstance[] data = new TimeSeriesInstance[dataset.length];
		for(int i = 0; i < dataset.length; i++){
			data[i] = dataset[i].truncateTo(ratio, minLength);
		}
		return new TimeSeriesSet(data);
	}
	
	public TimeSeriesSet truncateBigTo(int length){
		TimeSeriesInstance[] data = new TimeSeriesInstance[dataset.length];
		for(int i = 0; i < dataset.length; i++){
			data[i] = dataset[i].truncateBigTo(length);
		}
		return new TimeSeriesSet(data);
	}
	
	//所有实例的类别
	public double[] getAllLabels() {
		double[] labels = new double[dataset.length];
		for(int i = 0; i < dataset.length; i++) {
			labels[i] = dataset[i].label;
		}
		return labels;
	}
	
	public int getLabelIndex(double value){
		return Arrays.binarySearch(labelset, value) + 1;
	}
	
	public int size(){
		if (this.dataset == null)
			return 0;
		return this.dataset.length;
	}
	
	private int maxLength(){
		if (this.dataset == null)
			return 0;
		
		int max = 0;
		for(int i = 0; i < this.dataset.length; i++)
		{
			if(this.dataset[i].length() > max)
			{
				max = this.dataset[i].length();
			}
		}
		return max;
	}
	
	public int getMaxLength()
	{
		return maxLength;
	}
	
	private int minLength(){
		if (this.dataset == null)
			return 0;
		
		int min = Integer.MAX_VALUE;
		for(int i = 0; i < this.dataset.length; i++)
		{
			if(this.dataset[i].length() < min)
			{
				min = this.dataset[i].length();
			}
		}
		return min;
	}
	
	public int getMinLength()
	{
		return minLength;
	}
	
	private int averageLength(){
		if (this.dataset == null)
			return 0;
		
		int sum = 0;
		for(int i = 0; i < this.dataset.length; i++)
		{
			sum += this.dataset[i].length();
		}
		return sum / this.dataset.length;
	}
	
	public int getAverageLength()
	{
		return averageLength;
	}
	
	public int[] getLengths()
	{
		int []lengths = new int[dataset.length];
		for(int i = 0; i < dataset.length; i++)
		{
			lengths[i] = dataset[i].data.length;
		}
		//Arrays.sort(lengths);
		return lengths;
	}

	public Problem toLinearProblem(){
		Problem problem = new Problem();
		int length = this.size();
	    //problem.bias = 1;
		problem.l = dataset.length;           // number of training examples
	    problem.n = dataset[0].data.length;   // number of features
	    problem.x = new FeatureNode[length][];   // feature nodes
	    problem.y = new double[length];       // target values

	    for(int i = 0; i < length; i++)
	    {
	    	problem.x[i] = dataset[i].toLinearNode();
	    	problem.y[i] = dataset[i].label;
	    }      
	    return problem;
		
		
	}
	
	public TimeSeries[] toWEASEL()
	{
		TimeSeries.APPLY_Z_NORM = false;
		ArrayList<TimeSeries> samples = new ArrayList<>();
		
		for(int i = 0; i < dataset.length; i++)
		{
			TimeSeries ts = dataset[i].toWEASEL();
	        ts.norm();
	        samples.add(ts);
		}
		
        
        return samples.toArray(new TimeSeries[]{});
	}
	
	public void norm()
	{
		for(int i = 0; i < dataset.length; i++)
		{
			dataset[i].norm(true);
		}
	}
}
