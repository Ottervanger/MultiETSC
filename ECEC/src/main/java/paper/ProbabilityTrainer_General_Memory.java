package paper;

import java.io.File;
import java.io.FileWriter;
import java.io.Writer;
import java.util.ArrayList;
import java.util.Arrays;

import com.opencsv.CSVWriter;

import Classifiers.sfa.classification.Classifier.Predictions;
import Classifiers.sfa.classification.WEASELClassifier;

import de.bwaldvogel.liblinear.SolverType;

import DataStructures.ProbabilityInformation;
import DataStructures.ProbabilityInformationVary;
import DataStructures.ProbabilityInstance;
import DataStructures.ProbabilityResult;

import DataStructures.TimeSeriesInstance;
import DataStructures.TimeSeriesSet;

import Utilities.CrossValidation;

public class ProbabilityTrainer_General_Memory {
    TimeSeriesSet train_data;
    TimeSeriesSet test_data;
    
    public static int nClassifiers = 20;
    public static int nFolds = 5;
    public static long seed = 0;

    TimeSeriesSet[] train_cv;
    TimeSeriesSet[] test_cv;

    int[] tSteps;
    
    ProbabilityInformationVary probs_infor;
    
    int EqualLength = 0;
    int PercentageLength = 1;
    
    private void init(TimeSeriesSet train_data, TimeSeriesSet test_data) {
        // global WEASEL settings
        WEASELClassifier.lowerBounding = true;
        WEASELClassifier.solverType = SolverType.L2R_LR;

         
        this.train_data = train_data;
        this.test_data = test_data;
        
        train_cv = new TimeSeriesSet[nFolds];
        test_cv = new TimeSeriesSet[nFolds];

        probs_infor = new ProbabilityInformationVary(nClassifiers);
    }
    
    public void process(TimeSeriesSet train_data, TimeSeriesSet test_data, String result_dir) {
        train_data.shuffle(seed);
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
        tSteps = new int[nClassifiers];
        int step = train_data.getMaxLength() / nClassifiers;
        for(int i = 0; i < nClassifiers; i++) {
            int length = Math.max(minLen, (i+1)*step);
            if (i == nClassifiers - 1)
                length = Integer.MAX_VALUE;
            tSteps[i] = length;
        }
    }
    
    private void trainSlaverClassifiers() {
        for(int i = 0; i < nFolds; i++) {
            for(int t = 0; t < nClassifiers; t++) {
                WEASELClassifier classifier = new WEASELClassifier();
                classifier.fit(train_cv[i].truncateTo(tSteps[t]).toWEASEL());
                Predictions probs = classifier.predictProbabilities(test_cv[i].truncateTo(tSteps[t]).toWEASEL());
                
                for(int k = 0; k < probs.probabilities.length; k++) {
                    ProbabilityInstance instance = newProbabilityInstance(probs, k, test_cv[i].truncateTo(tSteps[t]).get(k), t);
                    probs_infor.getTrainGroup(t).add(instance);
                }
                
                System.out.printf("slaver %02d_%02d\r", i+1, t+1);
            }
        }
        System.out.printf("\n");
    }
    
    private void trainMasterClassifers() {
        for(int t = 0; t < nClassifiers; t++) {
            WEASELClassifier classifier = new WEASELClassifier();
            classifier.fit(train_data.truncateTo(tSteps[t]).toWEASEL());
            
            Predictions probs = classifier.predictProbabilities(test_data.truncateTo(tSteps[t]).toWEASEL());
            for(int k = 0; k < probs.probabilities.length; k++) {
                ProbabilityInstance instance = newProbabilityInstance(probs, k, test_data.truncateTo(tSteps[t]).get(k), t);
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

    private ProbabilityInstance newProbabilityInstance(Predictions result, int idx, TimeSeriesInstance ts, int groupID) 
    {
        ProbabilityInstance instance = new ProbabilityInstance();
        instance.groupIndex = groupID;
        instance.label = ts.getLabel();
        instance.index = ts.getIndex();
        instance.currentLength = ts.length();
        instance.fullLength = ts.getFullLength();
        instance.probs = new double[train_data.labelset.length];
        for(int i = 0; i < result.realLabels.length; i++) {
            double tmp_label = result.realLabels[i];
            int tmp_index = Arrays.binarySearch(train_data.labelset, tmp_label);
            instance.probs[tmp_index] = result.probabilities[idx][i];
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
    
    private void writeCSV(ArrayList<ProbabilityInstance> group, String file){
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

    public ProbabilityInformation asProbabilityInformation() {
        ProbabilityInformation infor = new ProbabilityInformation();
        int probs_index = 4;
        double probSum = 0.0;
        for(int i = 0; i < nClassifiers; i++) {
            ArrayList<ProbabilityInstance> group = probs_infor.getTrainGroup(i);

            if (i == 0) {
                infor.trainLabels = new double[group.size()];
                infor.trainLength = new int[group.size()];
                infor.trainStepLength = new int[group.size()][nClassifiers];
                infor.trainProbs = new double[group.size()][nClassifiers][group.get(0).probs.length];
            }

            int j = 0;
            for(ProbabilityInstance ins : group) {
                infor.trainLabels[j] = ins.label;
                infor.trainLength[j] = ins.fullLength;
                infor.trainStepLength[j][i] = ins.currentLength;
                infor.trainProbs[j][i] = ins.probs.clone();
                for(double p : ins.probs)
                    probSum += p * p;
                j++;
            }
        }
        System.out.printf("probs train: %10.3f\n", probSum);
        probSum = 0;

        for(int i = 0; i < nClassifiers; i++) {
            ArrayList<ProbabilityInstance> group = probs_infor.getTestGroup(i);

            if (i == 0) {
                infor.testLabels = new double[group.size()];
                infor.testLength = new int[group.size()];
                infor.testStepLength = new int[group.size()][nClassifiers];
                infor.testProbs = new double[group.size()][nClassifiers][group.get(0).probs.length];
            }

            int j = 0;
            for(ProbabilityInstance ins : group) {
                infor.testLabels[j] = ins.label;
                infor.testLength[j] = ins.fullLength;
                infor.testStepLength[j][i] = ins.currentLength;
                infor.testProbs[j][i] = ins.probs.clone();
                for(double p : ins.probs)
                    probSum += p * p;
                j++;
            }
        }
        System.out.printf("probs test:  %10.3f\n\n", probSum);
        infor.postprocess();
        return infor;
    }
}
