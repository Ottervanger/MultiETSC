package ECEC;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import java.util.Collections;

import com.carrotsearch.hppc.DoubleArrayList;
import sfa.timeseries.TimeSeries;

public class ECECClassifier {

    private double m_ratio = 0.8;

    private ProbabilityTrainer trainer;

    public class Result {
        public double accuracy;
        public double earliness;
        public double fcost;
    }

    public ECECClassifier() {
        trainer = new ProbabilityTrainer();
    }
    
    public Result fitAndTest(TimeSeries[] trainSamples, TimeSeries[] testSamples) {
        Probabilities probInfo = trainer.process(trainSamples, testSamples);

        double[][] confid = getConfidence(probInfo);
        double threshold = trainThreshold(probInfo, confid);

        Result result = test(probInfo, confid, threshold);
        
        return result;
    }

    public void setParameter(String name, String value) {
        switch(name) {
            case "ratio":
                m_ratio = Double.parseDouble(value);
                break;
            case "nClassifiers":
                trainer.nClassifiers = Integer.parseInt(value);
                break;
            case "nFolds":
                trainer.nFolds = Integer.parseInt(value);
                break;
            case "minLen":
                trainer.minLen = Integer.parseInt(value);
                break;
            case "maxLen":
                trainer.maxLen = Integer.parseInt(value);
                break;
            case "seed":
                trainer.seed = Long.parseLong(value);
                break;
        }
    }

    private class RuleResult {
        int step;
        int label;
    }

    private double[][] getConfidence(Probabilities probInfo) {
        double[][] confid = new double[probInfo.tSteps.length][probInfo.labelset.length];
        for(int i = 0; i < probInfo.tSteps.length; i++) {
            int[] nCorrect = new int[probInfo.labelset.length];
            int[] nPredicted = new int[probInfo.labelset.length];
            for(int j = 0; j < probInfo.train.labels.length; j++) {
                int max_index = argmax(probInfo.train.probs[j][i]);
                if (probInfo.labelset[max_index] == probInfo.train.labels[j]) {
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
        for (T[] l : a) {
            s.addAll(Arrays.asList(l));
        }
        ArrayList<T> r = new ArrayList<>(s);
        Collections.sort(r);
        return r;
    }

    private int argmax(double[] data) {
        int imax = 0;
        for(int i = 0; i < data.length; i++) {
            if(data[i] > data[imax]) {
                imax = i;
            }
        }
        return imax;
    }
    
    private double trainThreshold(Probabilities probInfo, double[][] confid) {
        int[][] predicted_labels = new int[probInfo.train.labels.length][probInfo.tSteps.length];
        Double[][] confidence = new Double[probInfo.train.labels.length][probInfo.tSteps.length];
        ArrayList<ArrayList<Double>> eachClass_confidence 
            = new ArrayList<ArrayList<Double>>(probInfo.labelset.length);
        for(int i = 0; i < probInfo.labelset.length; i++) {
            eachClass_confidence.add(new ArrayList<Double>());
        }
        
        for (int i = 0; i < probInfo.train.labels.length; i++) {
            for(int j = 0; j < probInfo.train.probs[i].length; j++) {
                int max = argmax(probInfo.train.probs[i][j]);
                predicted_labels[i][j] = max;
                double mod = 1;
                for(int k = 0; k <= j; k++) {
                    if (predicted_labels[i][k] == max) {
                        mod = mod * confid[k][max];
                    }
                }
                confidence[i][j] = 1 - mod;
                
                int realIndex = Arrays.binarySearch(probInfo.labelset, probInfo.train.labels[i]);
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
            for(int j = 0; j < probInfo.train.labels.length; j++) {
                for(int k = 0; k < probInfo.tSteps.length; k++) {
                    if(confidence[j][k] > threshold || k == probInfo.tSteps.length - 1) {
                        double ear = Math.min(1.0, ((double) probInfo.tSteps[k] / probInfo.train.length[j]));
                        earliness += ear;
                        if (probInfo.labelset[predicted_labels[j][k]] == probInfo.train.labels[j]) {
                            success++;
                        }
                        break;
                    }
                }
            }
            
            double cost = m_ratio * (probInfo.train.labels.length - success) + (1 - m_ratio) * earliness;
            if (cost < min) {
                min = cost;
                best_confidence = threshold;
            }
        }
        
        System.out.println("threshold:" + Double.toString(best_confidence));
        return best_confidence;
    }
    
    public Result test(Probabilities probInfo, double[][] confid, double threshold) {
        Result result = new Result();
        int instanceNum = probInfo.test.probs.length;
        int accuracyNum = 0;
        double earliness = 0;
        
        for (int i = 0; i < instanceNum; i++) {
            RuleResult ret = fuseConfid(probInfo.test.probs[i], confid, threshold);
            accuracyNum += (probInfo.labelset[ret.label] == probInfo.test.labels[i] ? 1 : 0);
            earliness += Math.min(1.0, ((double) probInfo.tSteps[ret.step-1] / probInfo.test.length[i]));     
        }
        
        result.accuracy = (double)accuracyNum /instanceNum;
        result.earliness = earliness /instanceNum;
        
        result.fcost = 1 / (m_ratio * (instanceNum - accuracyNum) + (1-m_ratio)*earliness);

        return result;
    }
    
    private RuleResult fuseConfid(double[][] probs, double[][] confid, double threshold) {
        RuleResult result = new RuleResult();
        int[] labels = new int[probs.length];
        
        for (int j = 0; j < probs.length; j++) {
            int max = argmax(probs[j]);
            labels[j] = max;
            double mod = 1;
            for (int k = 0; k <= j; k++) {
                if (labels[k] == max) {
                    mod = mod * confid[k][max];
                }
            }
            
            if (1 - mod >= threshold || j == probs.length - 1) {
                result.step = (j + 1);
                result.label = max;
                break;
            }
        }
        return result;
    }
}
