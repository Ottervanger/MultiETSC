package ECEC;

public class Probabilities {

	public class Element {
		public int[] length;
		public double[] labels;
		public double[][][] probs;
	}

	public int[] tSteps;
	public double[] labelset;
	
	public Element train;
	public Element test;
	
	public Probabilities() {
		train = new Element();
		test = new Element();
	}
}
