package paper;

import java.util.ArrayList;
import java.util.Arrays;

import com.carrotsearch.hppc.DoubleArrayList;

import DataStructures.EarlyClassifierResult;
import DataStructures.ProbabilityInformation;
import Utilities.StatisticalUtilities;

public class EarlyClassifier_ProbThreshold {

	
	private double m_threshold = 0.99;
	
	private double m_ratio = 0.8;
	
	class statisticsProb
	{
		int index;
		int step;
		double confidence;
		boolean correct;
		double label;
	}
	
	statisticsProb[] m_allProbs;
	ArrayList<ArrayList<statisticsProb>> m_eachClassProbs;
	
	class StepInfor
	{
		double label;
		int sampleNum;
		int precitedNum;
		int correct;
		double maxProb;
		ArrayList<double[]> probs;
		DoubleArrayList real_labels; 
		DoubleArrayList twoLabel;
	}
	
	StepInfor[][] m_stepInfo;

	public EarlyClassifier_ProbThreshold()
	{
	}
	///////////////////////////////////////////////////////
	
	private ProbabilityInformation probs_data;
	
	public class RuleResult
	{
		int step;
		int label;
	}
	
	public void buildClassifier(ProbabilityInformation data)
	{
		this.probs_data = data;
		
		train();
	}
	
	public EarlyClassifierResult predict(ProbabilityInformation data){
		this.probs_data = data;
		
		train();
		
		EarlyClassifierResult result = test(true); 
		
		return result;
	}
	
	private void train() 
	{
		trainModel();
		trainThreshold();
	}
	
	private void trainThreshold()
	{
		double[][][] testdata = probs_data.trainProbs;
		double[] labels = probs_data.trainLabels;
		int[] full_length = probs_data.trainLength;
		int[][] step_length = probs_data.trainStepLength;
		int instanceNum = testdata.length;
		
		int[][] predicted_labels = new int[instanceNum][probs_data.stepNum];
		double[][] confidence = new double[instanceNum][probs_data.stepNum];
		int pos = 0;
		double[] all_confidence = new double[instanceNum * probs_data.stepNum];
		ArrayList<ArrayList<Double>> eachClass_confidence 
			= new ArrayList<ArrayList<Double>>(probs_data.labelNum);
		for(int i = 0; i < probs_data.labelNum; i++)
		{
			eachClass_confidence.add(new ArrayList<Double>());
		}
		
		for (int i = 0; i < instanceNum; i++)
		{
			double[][] probs = testdata[i];
			
			for(int j = 0; j < probs.length; j++)
			{
				int max = StatisticalUtilities.maxIndex(probs[j]);
				predicted_labels[i][j] = max;
				double mod = 1;
				for(int k = 0; k <= j; k++)
				{
					if (predicted_labels[i][k] == max)
					{
						double correct = (double)m_stepInfo[k][max].correct / m_stepInfo[k][max].precitedNum;
						mod = mod * (1-correct);
					}
				}
				confidence[i][j] = 1 - mod;
				
				all_confidence[pos++] = confidence[i][j];
				int realIndex = Arrays.binarySearch(probs_data.labelTypes, labels[i]);
				eachClass_confidence.get(realIndex).add(confidence[i][j]);
			}	
		}

		Arrays.sort(all_confidence);
		
		ArrayList<Double> uniqueConfidence = new ArrayList<Double>();
		double currect = all_confidence[0];
		uniqueConfidence.add(currect);
		for(int i = 1; i < all_confidence.length; i++)
		{
			if (all_confidence[i] != currect)
			{
				currect = all_confidence[i];
				uniqueConfidence.add(currect);
			}
		}
		
		double[] middle_correct = new double[uniqueConfidence.size() - 1];
		double[] middle_earliness = new double[uniqueConfidence.size() - 1];
		double[] middle_cost = new double[uniqueConfidence.size() - 1];
		double[] middle = new double[uniqueConfidence.size() - 1];
		for(int i = 0; i < middle.length; i++)
		{
			middle[i] = (uniqueConfidence.get(i) + uniqueConfidence.get(i+1) ) / 2;
		}
		
		double min = Double.MAX_VALUE;
		double best_confidence = 0;
		for(int i = 0; i < middle.length; i++)
		{
			double threshold = middle[i];
			int success = 0;
			double earliness = 0;
			for(int j = 0; j < instanceNum; j++)
			{
				for(int k = 0; k < probs_data.stepNum; k++)
				{
					if(confidence[j][k] > threshold || k == probs_data.stepNum - 1)
					{
						double ear = Math.min(1.0, ((double) step_length[j][k] / full_length[j]));
						earliness += ear;
						if (probs_data.labelTypes[predicted_labels[j][k]] == labels[j])
						{
							success++;	
						}
						break;
					}
				}
			}
			
			double ratio = m_ratio;
			double cost = ratio * (instanceNum - success) + (1 - ratio) * earliness;
			middle_cost[i] = cost;
			middle_correct[i] = success;
			middle_earliness[i] = earliness;
			if(cost < min)
			{
				min = cost;
				best_confidence = threshold;
			}
		}
		
		m_threshold = best_confidence;
		System.out.println("threshold:" + Double.toString(m_threshold));
	}

	private void newStepInfor()
	{
		m_stepInfo = new StepInfor[this.probs_data.stepNum][probs_data.labelNum];
		for(int i = 0; i < this.probs_data.stepNum; i++)
		{
			for(int j = 0; j < probs_data.labelNum; j++)
			{
				m_stepInfo[i][j] = new StepInfor();
				m_stepInfo[i][j].label = probs_data.labelTypes[j];
				m_stepInfo[i][j].probs = new ArrayList<double[]>();
				m_stepInfo[i][j].real_labels = new DoubleArrayList(); 
				m_stepInfo[i][j].twoLabel = new DoubleArrayList(); 
			}
		}
	}
	
	private void trainModel()
	{
		newStepInfor();
		
		for(int i = 0; i < this.probs_data.stepNum; i++) 
		{
			ArrayList<double[]> correct_probs = new ArrayList<>();  //仅分类正确的样本
			DoubleArrayList correct_labels = new DoubleArrayList(); //仅分类正确的类别
			double[][] probs = new double[this.probs_data.trainNum][this.probs_data.labelNum];
			double[] labels = new double[this.probs_data.trainNum];
			for(int j = 0; j < this.probs_data.trainNum; j++) 
			{
				for(int k = 0; k < this.probs_data.labelNum; k++) 
				{
					probs[j][k] = this.probs_data.trainProbs[j][i][k];
				}
				
				int max_index = StatisticalUtilities.maxIndex(probs[j]);
				if (this.probs_data.labelTypes[max_index] == this.probs_data.trainLabels[j])
				{
					correct_probs.add(probs[j]);
					correct_labels.add(1);
					labels[j] = 1;
					
					m_stepInfo[i][max_index].correct++;
				}
				else
				{
					labels[j] = -1;
				}
				
				int realLabelIndex = Arrays.binarySearch(probs_data.labelTypes, probs_data.trainLabels[j]);
				m_stepInfo[i][realLabelIndex].sampleNum++;
				m_stepInfo[i][max_index].precitedNum++;
				m_stepInfo[i][max_index].maxProb = probs[j][max_index];
				m_stepInfo[i][max_index].probs.add(probs[j]);
				m_stepInfo[i][max_index].real_labels.add(probs_data.trainLabels[j]);
				m_stepInfo[i][max_index].twoLabel.add(labels[j]);
			}
		}
	}
	
	public EarlyClassifierResult test(boolean test)
	{
		double[][][] testdata;
		double [] labels;
		int[] full_length;
		int[][] step_length;
		//为true时验证测试数据，为false时验证训练数据
		if (test)
		{
			testdata = probs_data.testProbs;
			labels = probs_data.testLabels;
			full_length = probs_data.testLength;
			step_length = probs_data.testStepLength;
		}else
		{
			testdata = probs_data.trainProbs;
			labels = probs_data.trainLabels;
			full_length = probs_data.trainLength;
			step_length = probs_data.trainStepLength;
		}
		
		EarlyClassifierResult result = new EarlyClassifierResult();
		int instanceNum = testdata.length;
		int accuracyNum = 0;
		double earliness = 0;
		
		for (int i = 0; i < instanceNum; i++)
		{
			RuleResult ret = getResult_probs(testdata[i]);
			accuracyNum += (probs_data.labelTypes[ret.label] == labels[i] ? 1 : 0);
			earliness += Math.min(1.0, ((double) step_length[i][ret.step-1] / full_length[i]));		
		}
		
		result.accuracy = (double)accuracyNum /instanceNum;
		result.earliness = earliness /instanceNum;
		result.f1_score = StatisticalUtilities.f1Score(result.accuracy, 1-result.earliness);
		
		double ratio = 0.8;
		result.fcost = 1 / (ratio * (instanceNum - accuracyNum) + (1-ratio)*earliness);

		return result;
	}
	
	private RuleResult getResult_probs(double[][] probs)
	{
		RuleResult result = new RuleResult();
		int[] labels = new int[probs_data.stepNum];
		double[] confidence = new double[probs_data.stepNum];
		
		for(int j = 0; j < probs.length; j++)
		{
			int max = StatisticalUtilities.maxIndex(probs[j]);
			labels[j] = max;
			double mod = 1;
			for(int k = 0; k <= j; k++)
			{
				if (labels[k] == max)
				{
					double correct = (double)m_stepInfo[k][max].correct / m_stepInfo[k][max].precitedNum;
					mod = mod * (1-correct);
				}
			}
			confidence[j] = 1 - mod;
			
			if(confidence[j] >= m_threshold || j == probs.length - 1)
			{
				result.step = (j + 1);
				result.label = max;
				break;
			}
		}
		return result;
	}
}
