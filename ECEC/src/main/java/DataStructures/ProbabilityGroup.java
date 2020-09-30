package DataStructures;

import java.util.ArrayList;

public class ProbabilityGroup {
	private ArrayList<ProbabilityInstance> group;
	
	public ProbabilityGroup()
	{
		group = new ArrayList<ProbabilityInstance>();
	}
	
	public void add(ProbabilityInstance instance)
	{
		group.add(instance);
	}
	
	public int size()
	{
		return group.size();
	}
	
	public ProbabilityInstance get(int index)
	{
		return group.get(index);
	}
	
	public ArrayList<ProbabilityInstance> getByInstanceIndex(int index)
	{
		ArrayList<ProbabilityInstance> instances = new ArrayList<ProbabilityInstance>();
		for(int i = 0; i < group.size(); i++)
		{
			if(group.get(i).index == index)
			{
				instances.add(group.get(i));
			}
		}
		return instances;
	}
}
