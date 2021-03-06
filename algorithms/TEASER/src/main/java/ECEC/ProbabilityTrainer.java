package ECEC;

import java.io.File;
import java.io.FileWriter;
import java.io.Writer;
import java.util.*;
import java.util.stream.Collectors;

import sfa.classification.Classifier.Predictions;
import sfa.classification.WEASELClassifier;
import sfa.timeseries.TimeSeries;

import de.bwaldvogel.liblinear.SolverType;

public class ProbabilityTrainer {
    
    protected int nClassifiers = 20;
    protected int nFolds = 5;
    protected int minLen = 3;
    protected int maxLen = 250;
    protected long seed = 0;

    private HashMap<Double, Integer> labelIdx;

    private class ProbabilityInstance {
        public int currentLength;
        public int fullLength;
        public double label;
        public double[] probs;

        public ProbabilityInstance(Predictions result, int idx, int currentLength, int fullLength, double label) {
            this.currentLength = currentLength;
            this.fullLength = fullLength;
            this.label = label;
            this.probs = new double[result.realLabels.length];
            for(int i = 0; i < result.realLabels.length; i++)
                this.probs[labelIdx.get(Double.valueOf(result.realLabels[i]))] = result.probabilities[idx][i];
        }
    }
    
    public ProbabilityTrainer() {
        // global WEASEL settings
        WEASELClassifier.lowerBounding = true;
        WEASELClassifier.solverType = SolverType.L2R_LR;
    }
    
    public Probabilities process(TimeSeries[] dataTrain, TimeSeries[] dataTest) {
        shuffle(dataTrain, seed);
        double[] train_labels = getLabels(dataTrain);
        
        ArrayList<ArrayList<Integer>> cv = generateCV(train_labels, nFolds);

        subset(dataTrain, cv.get(0));

        int[] tSteps = generateStepData(minLen, getMax(dataTrain, maxLen));
        double[] labelset = getLabelSet(dataTrain);
        setLabelIdx(labelset);

        ArrayList<ProbabilityInstance>[] trainProbs = new ArrayList[nClassifiers];
        ArrayList<ProbabilityInstance>[] testProbs = new ArrayList[nClassifiers];
        for (int i = 0; i < nClassifiers; ++i) {
            trainProbs[i] = new ArrayList<ProbabilityInstance>();
            testProbs[i] = new ArrayList<ProbabilityInstance>();
        }
        trainSlaverClassifiers(dataTrain, cv, tSteps, trainProbs);
        trainMasterClassifers(dataTrain, dataTest, tSteps, testProbs);
        return asProbabilities(trainProbs, testProbs, tSteps, labelset);
    }

    private double[] getLabels(TimeSeries[] data) {
        double[] labels = new double[data.length];
        int i = 0;
        for (TimeSeries d : data)
            labels[i++] = d.getLabel();
        return labels;
    }

    private double[] getLabelSet(TimeSeries[] data) {
        ArrayList<Double> labels = new ArrayList<>();
        for (TimeSeries d : data)
            if(labels.indexOf(d.getLabel()) < 0)
                labels.add(d.getLabel());
        Collections.sort(labels);
        return labels.stream().mapToDouble(Double::doubleValue).toArray();
    }

    private int getMax(TimeSeries[] samples, int maxWindowSize) {
        int max = 0;
        for (TimeSeries ts : samples)
            max = Math.max(ts.getLength(), max);
        return Math.min(maxWindowSize, max);
    }

    private TimeSeries[] truncate(TimeSeries[] samples, int len) {
        ArrayList<TimeSeries> li = new ArrayList<TimeSeries>();
        for (TimeSeries ts : samples)
            li.add(ts.getSubsequence(0, len));
        return li.toArray(new TimeSeries[]{});
    }
    
    private int[] generateStepData(int minLen, int maxLen) {
        minLen = Math.max(3, minLen);
        int[] tSteps = new int[nClassifiers];
        double step = (maxLen - minLen) / (double)nClassifiers;
        for(int i = 0; i < nClassifiers; i++) {
            int length = (int) Math.round(step * i + minLen);
            if (i == nClassifiers - 1)
                length = maxLen;
            tSteps[i] = length;
        }
        return tSteps;
    }

    // Fisher–Yates shuffle
    public void shuffle(TimeSeries[] samples, long seed) {
        Random r = new Random(seed);
        for (int i = 0; i < samples.length - 1; i++) {
            int si = i + r.nextInt(samples.length - i);
            if (si > i) {
                TimeSeries tmp = samples[i];
                samples[i] = samples[si];
                samples[si] = tmp;
            }
        }
    }

    private void setLabelIdx(double[] labelset) {
        labelIdx = new HashMap<Double, Integer>();
        for (int i = 0; i < labelset.length; ++i)
            labelIdx.put(labelset[i], i);
    }

    private TimeSeries[] subset(TimeSeries[] samples, ArrayList<Integer> index){
        return index.stream().map(idx -> samples[idx]).toArray(TimeSeries[]::new);
    }
    
    private void trainSlaverClassifiers(TimeSeries[] dataTrain, ArrayList<ArrayList<Integer>> cv, int[] tSteps, ArrayList<ProbabilityInstance>[] trainProbs) {
        TimeSeries[][][] cv_data = makeCVSplit(dataTrain, cv);
        TimeSeries[][] train_cv = cv_data[0];
        TimeSeries[][] test_cv = cv_data[1];

        for(int i = 0; i < nFolds; i++) {
            for(int t = 0; t < nClassifiers; t++) {
                WEASELClassifier classifier = new WEASELClassifier();
                classifier.fit(truncate(train_cv[i], tSteps[t]));
                Predictions probs = classifier.predictProbabilities(truncate(test_cv[i], tSteps[t]));
                
                for(int k = 0; k < probs.probabilities.length; k++) {
                    ProbabilityInstance instance = new ProbabilityInstance(probs, k, tSteps[t], test_cv[i][k].getLength(), test_cv[i][k].getLabel());
                    trainProbs[t].add(instance);
                }
                
                System.out.printf("slaver %02d_%02d\r", i+1, t+1);
            }
        }
        System.out.printf("\n");
    }
    
    private void trainMasterClassifers(TimeSeries[] dataTrain, TimeSeries[] dataTest, int[] tSteps, ArrayList<ProbabilityInstance>[] testProbs) {
        for(int t = 0; t < nClassifiers; t++) {
            WEASELClassifier classifier = new WEASELClassifier();
            classifier.fit(truncate(dataTrain, tSteps[t]));
            
            Predictions probs = classifier.predictProbabilities(truncate(dataTest, tSteps[t]));
            for(int k = 0; k < probs.probabilities.length; k++) {
                ProbabilityInstance instance = new ProbabilityInstance(probs, k, tSteps[t], dataTest[k].getLength(), dataTest[k].getLabel());
                testProbs[t].add(instance);
            }
            System.out.printf("master %02d\r", t+1);
        }
        System.out.printf("\n");
    }

    private ArrayList<ArrayList<Integer>> generateCV(double[] labels, int fold){
        ArrayList<Double> uniqueLabels = new ArrayList<>();
        for(double label : labels)
            if(!uniqueLabels.contains(label))
                uniqueLabels.add(label);

        uniqueLabels.trimToSize();
        
        int[] index = new int[labels.length];
        int pos = 0;
        for(double currentLabel : uniqueLabels)
            for(int j = 0; j < labels.length; j++)
                if(labels[j] == currentLabel)
                    index[pos++] = j;
        
        ArrayList<ArrayList<Integer>> cv = new ArrayList<ArrayList<Integer>>(fold);
        for(int i = 0; i < fold; i++)
            cv.add(new ArrayList<Integer>());
        
        for(int i = 0; i < index.length; i++)
            cv.get(i % fold).add(index[i]);
        
        return cv;
    }

    private TimeSeries[][][] makeCVSplit(TimeSeries[] data, ArrayList<ArrayList<Integer>> cv) {
        TimeSeries[][] train_cv = new TimeSeries[cv.size()][];
        TimeSeries[][] test_cv = new TimeSeries[cv.size()][];

        if (cv.size() == 1) {
            train_cv[0] = subset(data, cv.get(0));
            test_cv[0] = subset(data, cv.get(0));
            return new TimeSeries[][][]{train_cv, test_cv};
        }
        
        for(int i = 0; i < cv.size(); i++){
            ArrayList<Integer> testIndex = cv.get(i);
            ArrayList<Integer> tranIndex = new ArrayList<Integer>();
            for(int j = 0; j < cv.size(); j++)
                if(i != j)
                    tranIndex.addAll(cv.get(j));
            train_cv[i] = subset(data, tranIndex);
            test_cv[i] = subset(data, testIndex);
        }
        return new TimeSeries[][][]{train_cv, test_cv};
    }

    private void fillElement(Probabilities.Element el, ArrayList<ProbabilityInstance>[] groups) {
        el.labels = new double[groups[0].size()];
        el.length = new int[groups[0].size()];
        el.probs = new double[groups[0].size()][nClassifiers][];
        for (int i = 0; i < nClassifiers; i++) {
            ArrayList<ProbabilityInstance> group = groups[i];
            int j = 0;
            for(ProbabilityInstance ins : group) {
                el.labels[j] = ins.label;
                el.length[j] = ins.fullLength;
                el.probs[j][i] = ins.probs.clone();
                j++;
            }
        }
    }
    
    private Probabilities asProbabilities(
            ArrayList<ProbabilityInstance>[] trainProbs,
            ArrayList<ProbabilityInstance>[] testProbs,
            int[] tSteps,
            double[] labelset) {
        Probabilities infor = new Probabilities();
        int probs_index = 4;
        infor.tSteps = tSteps.clone();
        fillElement(infor.train, trainProbs);
        fillElement(infor.test, testProbs);
        infor.labelset = labelset;
        return infor;
    }
}
