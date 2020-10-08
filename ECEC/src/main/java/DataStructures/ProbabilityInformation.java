package DataStructures;

import java.io.*;

public class ProbabilityInformation implements Serializable {

	public class Element implements Serializable {
		public int[] length;
		public double[] labels;
		public double[][][] probs;
	}

	public int[] tSteps;
	public double[] labelset;
	
	public Element train;
	public Element test;
	
	public ProbabilityInformation() {
		train = new Element();
		test = new Element();
	}

	public static ProbabilityInformation fromFile(String filename) throws IOException {
		try {
			FileInputStream f = new FileInputStream(new File(filename));
			ObjectInputStream o = new ObjectInputStream(f);
			ProbabilityInformation pri = (ProbabilityInformation) o.readObject();
			o.close();
			f.close();
			return pri;
		} catch (FileNotFoundException e) {
			System.out.println("File not found");
		} catch (IOException e) {
			System.out.println("Error initializing stream");
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
		}
		throw new IOException("Failed to load instance");
	}

	public void toFile(String filename) {
		try {
			FileOutputStream f = new FileOutputStream(new File(filename));
			ObjectOutputStream o = new ObjectOutputStream(f);
			o.writeObject(this);
			o.close();
			f.close();
		} catch (FileNotFoundException e) {
			System.out.println("File not found");
		} catch (IOException e) {
			System.out.println("Error initializing stream");
		}
	}
}
