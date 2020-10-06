package paper;

import java.io.File;
import java.io.FileWriter;
import java.io.Writer;
import java.util.ArrayList;
import java.util.Arrays;

import com.opencsv.CSVWriter;

import Classifiers.sfa.classification.Classifier.Predictions;
import Classifiers.sfa.classification.WEASELClassifier;
import Classifiers.sfa.timeseries.TimeSeries;

import de.bwaldvogel.liblinear.SolverType;

import DataStructures.ProbabilityInformation;
import DataStructures.ProbabilityInformationVary;
import DataStructures.ProbabilityInstance;
import DataStructures.ProbabilityResult;

import DataStructures.TimeSeriesInstance;
import DataStructures.TimeSeriesSet;

import Utilities.CrossValidation;

public class ProbabilityTrainer_General_Memory {
    
    public static int nClassifiers = 20;
    public static int nFolds = 5;
    public static long seed = 0;
    
    public ProbabilityTrainer_General_Memory() {
        // global WEASEL settings
        WEASELClassifier.lowerBounding = true;
        WEASELClassifier.solverType = SolverType.L2R_LR;
    }
    
    public ProbabilityInformation process(TimeSeriesSet train_data, TimeSeriesSet test_data, String result_dir) {
        train_data.shuffle(seed);
        double[] train_labels = getLabels(train_data.toWEASEL());
        
        ArrayList<ArrayList<Integer>> cv = CrossValidation.generateCV(train_labels, nFolds);

        int minLen = 3;
        int maxLen = train_data.getMaxLength();
        int[] tSteps = generateStepData(minLen, maxLen);

        ProbabilityInformationVary pInfoVar = new ProbabilityInformationVary(nClassifiers);
        trainSlaverClassifiers(train_data, cv, tSteps, pInfoVar);
        trainMasterClassifers(train_data, test_data, tSteps, pInfoVar);
        return asProbabilityInformation(pInfoVar);
    }

    private double[] getLabels(TimeSeries[] data) {
        double[] labels = new double[data.length];
        int i = 0;
        for (TimeSeries d : data)
            labels[i++] = d.getLabel();
        return labels;
    }
    
    private int[] generateStepData(int minLen, int maxLen) {
        int[] tSteps = new int[nClassifiers];
        int step = maxLen / nClassifiers;
        for(int i = 0; i < nClassifiers; i++) {
            int length = Math.max(minLen, (i+1)*step);
            if (i == nClassifiers - 1)
                length = Integer.MAX_VALUE;
            tSteps[i] = length;
        }
        return tSteps;
    }
    
    private void trainSlaverClassifiers(TimeSeriesSet train_data, ArrayList<ArrayList<Integer>> cv, int[] tSteps, ProbabilityInformationVary pInfoVar) {
        TimeSeriesSet[][] cv_data = divideMultiDataset(train_data, cv);
        TimeSeriesSet[] train_cv = cv_data[0];
        TimeSeriesSet[] test_cv = cv_data[1];

        for(int i = 0; i < nFolds; i++) {
            for(int t = 0; t < nClassifiers; t++) {
                WEASELClassifier classifier = new WEASELClassifier();
                classifier.fit(train_cv[i].toWEASEL(tSteps[t]));
                Predictions probs = classifier.predictProbabilities(test_cv[i].toWEASEL(tSteps[t]));
                
                for(int k = 0; k < probs.probabilities.length; k++) {
                    ProbabilityInstance instance = newProbabilityInstance(probs, k, test_cv[i].get(k).truncateTo(tSteps[t]), t, train_data.labelset);
                    pInfoVar.getTrainGroup(t).add(instance);
                }
                
                System.out.printf("slaver %02d_%02d\r", i+1, t+1);
            }
        }
        System.out.printf("\n");
    }
    
    private void trainMasterClassifers(TimeSeriesSet train_data, TimeSeriesSet test_data, int[] tSteps, ProbabilityInformationVary pInfoVar) {
        for(int t = 0; t < nClassifiers; t++) {
            WEASELClassifier classifier = new WEASELClassifier();
            classifier.fit(train_data.toWEASEL(tSteps[t]));
            
            Predictions probs = classifier.predictProbabilities(test_data.toWEASEL(tSteps[t]));
            for(int k = 0; k < probs.probabilities.length; k++) {
                ProbabilityInstance instance = newProbabilityInstance(probs, k, test_data.get(k).truncateTo(tSteps[t]), t, train_data.labelset);
                pInfoVar.getTestGroup(t).add(instance);
            }
            System.out.printf("master %02d\r", t+1);
        }
        System.out.printf("\n");
    }

    private TimeSeriesSet[][] divideMultiDataset(TimeSeriesSet data, ArrayList<ArrayList<Integer>> cv) {
        TimeSeriesSet[] train_cv = new TimeSeriesSet[nFolds];
        TimeSeriesSet[] test_cv = new TimeSeriesSet[nFolds];

        if (cv.size() == 1) {
            train_cv[0] = data.subset(cv.get(0));
            test_cv[0] = data.subset(cv.get(0));
            return new TimeSeriesSet[][]{train_cv, test_cv};
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
        return new TimeSeriesSet[][]{train_cv, test_cv};
    }

    private ProbabilityInstance newProbabilityInstance(Predictions result, int idx, TimeSeriesInstance ts, int groupID, double[] labelset) {
        ProbabilityInstance instance = new ProbabilityInstance();
        instance.groupIndex = groupID;
        instance.label = ts.getLabel();
        instance.index = ts.getIndex();
        instance.currentLength = ts.length();
        instance.fullLength = ts.getFullLength();
        instance.probs = new double[labelset.length];
        for(int i = 0; i < result.realLabels.length; i++) {
            double tmp_label = result.realLabels[i];
            int tmp_index = Arrays.binarySearch(labelset, tmp_label);
            instance.probs[tmp_index] = result.probabilities[idx][i];
        }
        return instance;
    }
    
    private ProbabilityInformation asProbabilityInformation(ProbabilityInformationVary pInfoVar) {
        ProbabilityInformation infor = new ProbabilityInformation();
        int probs_index = 4;
        double probSum = 0.0;
        for(int i = 0; i < nClassifiers; i++) {
            ArrayList<ProbabilityInstance> group = pInfoVar.getTrainGroup(i);

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
            ArrayList<ProbabilityInstance> group = pInfoVar.getTestGroup(i);

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
