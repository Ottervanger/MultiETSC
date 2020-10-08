package paper;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import java.util.Collections;

import com.carrotsearch.hppc.DoubleArrayList;

import DataStructures.ProbabilityInformation;
import Utilities.StatisticalUtilities;

public class EarlyClassifier_ProbThreshold {

    public class RuleResult {
        int step;
        int label;
    }

    public class Result {
        public double accuracy;
        public double earliness;
        public double fcost;
    }
    
    private double m_ratio = 0.8;
    
    private ProbabilityInformation probs_data;
    
    public Result predict(ProbabilityInformation data) {
        this.probs_data = data;
        double[][] confid = getConfidence();
        double threshold = trainThreshold(confid);

        Result result = test(confid, threshold); 
        
        return result;
    }

    private double[][] getConfidence() {
        double[][] confid = new double[probs_data.stepNum][probs_data.labelNum];
        for(int i = 0; i < probs_data.stepNum; i++) {
            int[] nCorrect = new int[probs_data.labelNum];
            int[] nPredicted = new int[probs_data.labelNum];
            for(int j = 0; j < probs_data.trainNum; j++) {
                int max_index = StatisticalUtilities.maxIndex(probs_data.trainProbs[j][i]);
                if (probs_data.labelTypes[max_index] == probs_data.trainLabels[j]) {
                    nCorrect[max_index]++;
                }
                nPredicted[max_index]++;
            }
            for (int k = 0; k < nPredicted.length; ++k)
                confid[i][k] = 1. - (double)nCorrect[k] / nPredicted[k];
        }
        return confid;
    }

    private <T extends Comparable<? super T>> ArrayList<T> unique(T[][] a) {
        Set<T> s = new HashSet<>();
        for (T[] l : a)
            s.addAll(Arrays.asList(l));
        ArrayList<T> r = new ArrayList<>(s);
        Collections.sort(r);
        return r;
    }
    
    private double trainThreshold(double[][] confid) {
        int[][] predicted_labels = new int[probs_data.trainNum][probs_data.stepNum];
        Double[][] confidence = new Double[probs_data.trainNum][probs_data.stepNum];
        ArrayList<ArrayList<Double>> eachClass_confidence 
            = new ArrayList<ArrayList<Double>>(probs_data.labelNum);
        for(int i = 0; i < probs_data.labelNum; i++) {
            eachClass_confidence.add(new ArrayList<Double>());
        }
        
        for (int i = 0; i < probs_data.trainNum; i++) {
            for(int j = 0; j < probs_data.trainProbs[i].length; j++) {
                int max = StatisticalUtilities.maxIndex(probs_data.trainProbs[i][j]);
                predicted_labels[i][j] = max;
                double mod = 1;
                for(int k = 0; k <= j; k++) {
                    if (predicted_labels[i][k] == max) {
                        mod = mod * confid[k][max];
                    }
                }
                confidence[i][j] = 1 - mod;
                
                int realIndex = Arrays.binarySearch(probs_data.labelTypes, probs_data.trainLabels[i]);
                eachClass_confidence.get(realIndex).add(confidence[i][j]);
            }   
        }

        ArrayList<Double> uniqueConfidence = unique(confidence);
        
        double[] middle = new double[uniqueConfidence.size() - 1];
        for(int i = 0; i < middle.length; i++) {
            middle[i] = (uniqueConfidence.get(i) + uniqueConfidence.get(i+1) ) / 2;
        }
        
        double min = Double.MAX_VALUE;
        double best_confidence = 0;
        for(int i = 0; i < middle.length; i++) {
            double threshold = middle[i];
            int success = 0;
            double earliness = 0;
            for(int j = 0; j < probs_data.trainNum; j++) {
                for(int k = 0; k < probs_data.stepNum; k++) {
                    if(confidence[j][k] > threshold || k == probs_data.stepNum - 1) {
                        double ear = Math.min(1.0, ((double) probs_data.trainStepLength[j][k] / probs_data.trainLength[j]));
                        earliness += ear;
                        if (probs_data.labelTypes[predicted_labels[j][k]] == probs_data.trainLabels[j]) {
                            success++;
                        }
                        break;
                    }
                }
            }
            
            double cost = m_ratio * (probs_data.trainNum - success) + (1 - m_ratio) * earliness;
            if(cost < min) {
                min = cost;
                best_confidence = threshold;
            }
        }
        
        System.out.println("threshold:" + Double.toString(best_confidence));
        return best_confidence;
    }
    
    public Result test(double[][] confid, double threshold) {
        Result result = new Result();
        int instanceNum = probs_data.testProbs.length;
        int accuracyNum = 0;
        double earliness = 0;
        
        for (int i = 0; i < instanceNum; i++) {
            RuleResult ret = fuseConfid(probs_data.testProbs[i], confid, threshold);
            accuracyNum += (probs_data.labelTypes[ret.label] == probs_data.testLabels[i] ? 1 : 0);
            earliness += Math.min(1.0, ((double) probs_data.testStepLength[i][ret.step-1] / probs_data.testLength[i]));     
        }
        
        result.accuracy = (double)accuracyNum /instanceNum;
        result.earliness = earliness /instanceNum;
        
        result.fcost = 1 / (m_ratio * (instanceNum - accuracyNum) + (1-m_ratio)*earliness);

        return result;
    }
    
    private RuleResult fuseConfid(double[][] probs, double[][] confid, double threshold) {
        RuleResult result = new RuleResult();
        int[] labels = new int[probs.length];
        
        for(int j = 0; j < probs.length; j++) {
            int max = StatisticalUtilities.maxIndex(probs[j]);
            labels[j] = max;
            double mod = 1;
            for(int k = 0; k <= j; k++) {
                if (labels[k] == max) {
                    mod = mod * confid[k][max];
                }
            }
            
            if(1 - mod >= threshold || j == probs.length - 1) {
                result.step = (j + 1);
                result.label = max;
                break;
            }
        }
        return result;
    }
}
