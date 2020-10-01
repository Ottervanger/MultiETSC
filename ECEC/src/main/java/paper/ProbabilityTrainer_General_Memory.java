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

import Utilities.CrossValidation;

public class ProbabilityTrainer_General_Memory {
    TimeSeriesSet train_data;
    TimeSeriesSet test_data;
    
    int nClassifiers;
    int nFolds;
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
    
    private void init(TimeSeriesSet train_data, TimeSeriesSet test_data){
        m_classifierID = 1;
         
        this.train_data = train_data;
        this.test_data = test_data;
        nFolds = 5;
        
        nClassifiers = 20;
        
        train_cv = new TimeSeriesSet[nFolds];
        test_cv = new TimeSeriesSet[nFolds];
        
        slaver_traindata = new TimeSeriesSet[nFolds][nClassifiers];
        slaver_testdata = new TimeSeriesSet[nFolds][nClassifiers];
        master_traindata = new TimeSeriesSet[nClassifiers];
        master_testdata = new TimeSeriesSet[nClassifiers];

        
        probs_infor = new ProbabilityInformationVary(nClassifiers);
        
        master_classifiers = new WEASEL[nClassifiers];
        for(int i = 0; i < nClassifiers; i++) {
            master_classifiers[i] = new WEASEL(m_classifierID++);
        }
    }
    
    public void process(TimeSeriesSet train_data, TimeSeriesSet test_data, String result_dir){
        init(train_data, test_data);
        double[] train_labels = train_data.getAllLabels();
        ArrayList<ArrayList<Integer>> cv = CrossValidation.generateCV(train_labels, nFolds);
        divideMultiDataset(cv, train_data);

        int minLen = 3;
        generateStepData(minLen);
        trainSlaverClassifiers();
        writeToFile_Slaver(result_dir);
        trainMasterClassifers();
        writeToFile_Master(result_dir);
    }
    
    private void generateStepData(int minLen) {
        int step = train_data.getMaxLength() / nClassifiers;
        for(int i = 0; i < nClassifiers; i++) {
            int length = Math.max(minLen, (i+1)*step);
            if (i == nClassifiers - 1)
                length = Integer.MAX_VALUE;
            master_traindata[i] = train_data.truncateTo(length);
            master_testdata[i] = test_data.truncateTo(length);
            for(int j = 0; j < nFolds; j++) {
                slaver_traindata[j][i] = train_cv[j].truncateTo(length);
                slaver_testdata[j][i] = test_cv[j].truncateTo(length);
            }
        }
    }
    
    private void trainSlaverClassifiers() {
        for(int i = 0; i < nFolds; i++) {
            for(int t = 0; t < nClassifiers; t++) {
                Classifier slaver_classifier = new WEASEL(m_classifierID++);
                slaver_classifier.buildClassifier(slaver_traindata[i][t]);
                ProbabilityResult[] probs = slaver_classifier.probabilityForInstance(slaver_testdata[i][t]);
                
                for(int k = 0; k < probs.length; k++) {
                    ProbabilityInstance instance = newProbabilityInstance(probs[k], slaver_testdata[i][t].get(k), t);
                    probs_infor.getTrainGroup(t).add(instance);
                }
                
                System.out.printf("slaver %02d_%02d\r", i+1, t+1);
            }
        }
        System.out.printf("\n");
    }
    
    private void trainMasterClassifers() {
        for(int t = 0; t < nClassifiers; t++) {
            master_classifiers[t].buildClassifier(master_traindata[t]);
            
            ProbabilityResult[] probs = master_classifiers[t].probabilityForInstance(master_testdata[t]);
            for(int k = 0; k < probs.length; k++) {
                ProbabilityInstance instance = newProbabilityInstance(probs[k], master_testdata[t].get(k), t);
                probs_infor.getTestGroup(t).add(instance);
            }
            System.out.printf("master %02d\r", t+1);
        }
        System.out.printf("\n");
    }

    private void divideMultiDataset(ArrayList<ArrayList<Integer>> cv, TimeSeriesSet data) {
        if (cv.size() == 1) {
            train_cv[0] = data.subset(cv.get(0));
            test_cv[0] = data.subset(cv.get(0));
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
            train_cv[i] = data.subset(tranIndex);
            test_cv[i] = data.subset(testIndex);
        }
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
    
    private void writeToFile_Slaver(String dir) {
        //train file
        for(int i = 0; i < nClassifiers; i++) {
            String train_file = dir + File.separator + "general-train-probs-" + Integer.toString(i+1) + ".csv";
            writeCSV(probs_infor.getTrainGroup(i), train_file);
        }
    }
    private void writeToFile_Master(String dir) {
        //test file
        for(int i = 0; i < nClassifiers; i++) {
            String test_file = dir + File.separator + "general-test-probs-" + Integer.toString(i+1) + ".csv";
            writeCSV(probs_infor.getTestGroup(i), test_file);
        }
    }
    
    private void writeCSV(ProbabilityGroup group, String file){
        try {
            Writer writer = new FileWriter(file);  
            CSVWriter csvWriter = new CSVWriter(writer); 
            
            int probs_index = 4;
            int len = group.get(0).probs.length + probs_index;
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
