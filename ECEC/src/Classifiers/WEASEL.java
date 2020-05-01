package Classifiers;

import Classifiers.sfa.classification.Classifier.Predictions;
import Classifiers.sfa.classification.Classifier.Score;
import Classifiers.sfa.classification.WEASELClassifier;
import Classifiers.sfa.classification.WEASELClassifier.WEASELModel;
import Classifiers.sfa.timeseries.TimeSeries;
import DataStructures.ProbabilityResult;
import DataStructures.TimeSeriesInstance;
import DataStructures.TimeSeriesSet;
import de.bwaldvogel.liblinear.SolverType;

public class WEASEL extends Classifier{

	private TimeSeriesSet m_traindata;
	private WEASELClassifier model;
	
	private int m_classifierID;
	
	public WEASEL(int index)
	{
		m_classifierID = index;
		model = new WEASELClassifier();
		WEASELClassifier.lowerBounding = true;
	    WEASELClassifier.solverType = SolverType.L2R_LR;
	}
	
	public void buildClassifier(TimeSeriesSet train){
		m_traindata = train;
		Score score = model.fit(train.toWEASEL());
	}
	
	public ProbabilityResult probabilityForInstance(TimeSeriesInstance instance){
		TimeSeries[] ts = new TimeSeries[1];
		ts[0] = instance.toWEASEL();
		Predictions probs = model.predictProbabilities(ts);
		
		ProbabilityResult result = new ProbabilityResult(probs.realLabels.length);
		for(int i = 0; i < probs.realLabels.length; i++)
		{
			result.labels[i] = probs.realLabels[i];
			result.probs[i] = probs.probabilities[0][i];
		}
		
		return result;
	}
	
	public ProbabilityResult[] probabilityForInstance(TimeSeriesSet test)
	{
		ProbabilityResult[] results = new ProbabilityResult[test.size()];
		for(int i = 0; i < test.size(); i++) 
		{
			results[i] = probabilityForInstance(test.get(i));
		}
		
		return results;
	}
	
	public double getLabel(TimeSeriesInstance instance){
		return -1;   //TODO
	}
}
