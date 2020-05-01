package Classifiers;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import DataStructures.EarlyClassifierResult;
import DataStructures.ProbabilityResult;
import DataStructures.TimeSeriesInstance;
import DataStructures.TimeSeriesSet;

public abstract class Classifier {
	
	protected int minLength;
	protected int maxLength;
	
	public abstract void buildClassifier(TimeSeriesSet train_data);
	
	public double getLabel(TimeSeriesInstance instance){
		return -1;   //TODO
	}
	
	public EarlyClassifierResult predict(TimeSeriesSet test)
	{
		return new EarlyClassifierResult();
	}
	
	public abstract ProbabilityResult probabilityForInstance(TimeSeriesInstance instance);
	
	public ProbabilityResult[] probabilityForInstance(TimeSeriesSet testdata)
	{
		return new ProbabilityResult[testdata.size()];
	}
	
	public double bestDistance(TimeSeriesInstance instance, double threshold)
	{
		return Double.MAX_VALUE;
	}
	
	public int minLength()
	{
		return minLength;
	}
	
	public int maxLength()
	{
		return maxLength;
	}
}
