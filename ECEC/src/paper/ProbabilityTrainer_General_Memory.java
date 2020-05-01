package paper;

import java.io.File;
import java.io.FileWriter;
import java.io.Writer;
import java.util.ArrayList;
import java.util.Arrays;

import com.opencsv.CSVWriter;

import Classifiers.Classifier;
import Classifiers.WEASEL;
import DataStructures.ProbabilityGroup;
import DataStructures.ProbabilityInformationVary;
import DataStructures.ProbabilityInstance;
import DataStructures.ProbabilityResult;
import DataStructures.TimeSeriesInstance;
import DataStructures.TimeSeriesSet;

public class ProbabilityTrainer_General_Memory {
	TimeSeriesSet train_data;
	TimeSeriesSet test_data;
	String dir;
	
	int min_length_threshold;
	int max_step_number;  //
	int step_length;   //步长
	int cv_number;     //几折交叉验证
	TimeSeriesSet[] train_cv;
	TimeSeriesSet[] test_cv;
	
	TimeSeriesSet[][] slaver_traindata;
	TimeSeriesSet[][] slaver_testdata;
	TimeSeriesSet[] master_traindata;
	TimeSeriesSet[] master_testdata;
	Classifier[] master_classifiers;
	
	ProbabilityInformationVary probs_infor;
	
	private int m_classifierID;
	
	int EqualLength = 0;
	int PercentageLength = 1;
	
	private void initClassifiers()
	{
		master_classifiers = new WEASEL[max_step_number];
		for(int i = 0; i < max_step_number; i++)
		{
			master_classifiers[i] = new WEASEL(m_classifierID++);
		}		
	}
	
	private void init(TimeSeriesSet train_data, TimeSeriesSet test_data, String result_dir){
		m_classifierID = 1;
		 
		this.train_data = train_data;
		this.test_data = test_data;
		this.dir = result_dir;
		min_length_threshold = 3;
		cv_number = 5;
		
		max_step_number = 20;
		
		step_length = train_data.getMaxLength() / 20;    //TODO
//		step_length = 12;    //TODO
		
		train_cv = new TimeSeriesSet[cv_number];
		test_cv = new TimeSeriesSet[cv_number];
		
		slaver_traindata = new TimeSeriesSet[cv_number][max_step_number];
		slaver_testdata = new TimeSeriesSet[cv_number][max_step_number];
		master_traindata = new TimeSeriesSet[max_step_number];
		master_testdata = new TimeSeriesSet[max_step_number];

		
		probs_infor = new ProbabilityInformationVary(max_step_number);
		
		initClassifiers();
	}
	
	public void process(TimeSeriesSet train_data, TimeSeriesSet test_data, String result_dir){
		init(train_data, test_data, result_dir);
		double[] train_labels = getTrainDataLabel();
		ArrayList<ArrayList<Integer>> cv = Utilities.CrossValidation.generateCV(train_labels, cv_number);
		divideMultiDataset(cv);
		
		generateStepData();
		System.out.println("prepare data over.");
		trainSlaverClassifiers();
		writeToFile_Slaver();
		trainMasterClassifers();
		writeToFile_Master();
	}
	
	private void generateEqualLengthSlaverData()
	{
		for(int i = 0; i < cv_number; i++)
		{
			for(int j = 0; j < max_step_number; j++)
			{
				int length = Math.max(min_length_threshold, (j+1)*step_length);
				if (j == max_step_number - 1) 
				{
					length = Integer.MAX_VALUE;
				}
				slaver_traindata[i][j] = train_cv[i].truncateTo(length);
				slaver_testdata[i][j] = test_cv[i].truncateTo(length);
			}
		}
	}
	
	private void generateEqualLengthMasterData()
	{
		for(int i = 0; i < max_step_number; i++) 
		{
			int length = Math.max(min_length_threshold, (i+1)*step_length);
			if (i == max_step_number - 1) 
			{
				length = Integer.MAX_VALUE;
			}
			master_traindata[i] = train_data.truncateTo(length);
			master_testdata[i] = test_data.truncateTo(length);
		}
	}
	
	private void generateStepData()
	{
		generateEqualLengthSlaverData();
		generateEqualLengthMasterData();	
	}
	
	private void trainSlaverClassifiers()
	{
		for(int i = 0; i < cv_number; i++)
		{
			for(int t = 0; t < max_step_number; t++)
			{
				Classifier slaver_classifier = new WEASEL(m_classifierID++);
				slaver_classifier.buildClassifier(slaver_traindata[i][t]);
				ProbabilityResult[] probs = slaver_classifier.probabilityForInstance(slaver_testdata[i][t]);
				
				for(int k = 0; k < probs.length; k++)
				{
					ProbabilityInstance instance = newProbabilityInstance(probs[k], slaver_testdata[i][t].get(k), t);
					probs_infor.getTrainGroup(t).add(instance);
				}
				
				System.out.println("slaver " + Integer.toString(i+1) + "_" + Integer.toString(t+1));
			}
		}
	}
	
	private void trainMasterClassifers()
	{
		for(int t = 0; t < max_step_number; t++)
		{
			{
				master_classifiers[t].buildClassifier(master_traindata[t]);
			}
			
			ProbabilityResult[] probs = master_classifiers[t].probabilityForInstance(master_testdata[t]);
			for(int k = 0; k < probs.length; k++)
			{
				ProbabilityInstance instance = newProbabilityInstance(probs[k], master_testdata[t].get(k), t);
				probs_infor.getTestGroup(t).add(instance);
			}
			System.out.println("master " + Integer.toString(t+1));
		}
	}

	private void divideMultiDataset(ArrayList<ArrayList<Integer>> cv){
		if (cv.size() == 1)
		{
			train_cv[0] = train_data.subset(cv.get(0));
			test_cv[0] = train_data.subset(cv.get(0));
			return;
		}
		
		for(int i = 0; i < cv.size(); i++){
			ArrayList<Integer> testIndex = cv.get(i);
			ArrayList<Integer> tranIndex = new ArrayList<Integer>();
			for(int j = 0; j < cv.size(); j++){
				if(i == j)
					continue;
				tranIndex.addAll(cv.get(j));
			}
			train_cv[i] = train_data.subset(tranIndex);
			test_cv[i] = train_data.subset(testIndex);
		}
	}
	
	private double[] getTrainDataLabel(){
		double[] labels = new double[train_data.size()];
		for(int i = 0; i < train_data.size(); i++){
			labels[i] = train_data.dataset[i].label;
		}
		
		return labels;
	}

	private ProbabilityInstance newProbabilityInstance(ProbabilityResult result, TimeSeriesInstance ts, int groupID) 
	{
		ProbabilityInstance instance = new ProbabilityInstance();
		instance.groupIndex = groupID;
		instance.label = ts.getLabel();
		instance.index = ts.getIndex();
		instance.currentLength = ts.length();
		instance.fullLength = ts.getFullLength();
		instance.probs = new double[train_data.labelset.length];
		for(int i = 0; i < result.labels.length; i++) {
			double tmp_label = result.labels[i];
			int tmp_index = Arrays.binarySearch(train_data.labelset, tmp_label);
			instance.probs[tmp_index] = result.probs[i];
		}
		return instance;
	}
	
	private void writeToFile_Slaver() {
		//train file
		for(int i = 0; i < max_step_number; i++) {
			String train_file = dir + File.separator + "general-train-probs-" + Integer.toString(i+1) + ".csv";
			writeCSV(probs_infor.getTrainGroup(i), train_file);
		}
	}
	private void writeToFile_Master() {
		//test file
		for(int i = 0; i < max_step_number; i++) {
			String test_file = dir + File.separator + "general-test-probs-" + Integer.toString(i+1) + ".csv";
			writeCSV(probs_infor.getTestGroup(i), test_file);
		}
	}
	
	private void writeCSV(ProbabilityGroup group, String file){
		try {
			Writer writer = new FileWriter(file);  
		    CSVWriter csvWriter = new CSVWriter(writer); 
		    
		    int probs_index = 4;
		    int len = group.get(0).probs.length + probs_index;  //前四列为 样本index，类别，样本长度，当前长度
			int instanceNumber = group.size();
			for(int i = 0; i < instanceNumber; i++) {
				String[] strs = new String[len];
				strs[0] = Integer.toString(group.get(i).index);
				strs[1] = Double.toString(group.get(i).label);
				strs[2] = Integer.toString(group.get(i).fullLength);
				strs[3] = Integer.toString(group.get(i).currentLength);
				for(int j = probs_index; j < len; j++) {
					strs[j] = Double.toString(group.get(i).probs[j - probs_index]);
				}
				csvWriter.writeNext(strs);
			}
			
			csvWriter.close(); 
		}catch(Exception e){
			e.printStackTrace();
		}
	}
}
