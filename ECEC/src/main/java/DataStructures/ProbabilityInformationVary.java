package DataStructures;

public class ProbabilityInformationVary {
	
	private ProbabilityGroup[] trainProbs;
	private ProbabilityGroup[] testProbs;
	
	private double[] labelset;
	private int trainNum;
	private int testNum;
	
	public ProbabilityInformationVary(int groups)
	{
		trainProbs = new ProbabilityGroup[groups];
		testProbs = new ProbabilityGroup[groups];
		for(int i = 0; i < groups; i++)
		{
			trainProbs[i] = new ProbabilityGroup();
			testProbs[i] = new ProbabilityGroup();
		}
	}
	
	public ProbabilityGroup[] getTrainGroup()
	{
		return trainProbs;
	}
	
	public ProbabilityGroup getTrainGroup(int index)
	{
		return trainProbs[index];
	}
	
	public ProbabilityGroup[] getTestGroup()
	{
		return testProbs;
	}
	
	public ProbabilityGroup getTestGroup(int index)
	{
		return testProbs[index];
	}
	
	public int getGroupNum()
	{
		return trainProbs.length;
	}
	
	public int getTrainInstanceNum()
	{
		return trainNum;
	}
	
	public void setTrainInstanceNum(int size)
	{
		trainNum = size;
	}
	
	public int getTestInstanceNum()
	{
		return testNum;
	}
	
	public void setTestInstanceNum(int size)
	{
		testNum = size;
	}
	
	public double[] getLabelset()
	{
		return labelset;
	}
	
	public void setLabelset(double[] labels)
	{
		labelset = labels.clone();
	}
}
