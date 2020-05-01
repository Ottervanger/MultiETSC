package DataStructures;

import java.util.Arrays;

import Classifiers.sfa.timeseries.TimeSeries;
import de.bwaldvogel.liblinear.FeatureNode;

public class TimeSeriesInstance {

	public double[] data = null;
	
	public double label = -999;
	
	private int m_index = -1;
	
	private int m_fullLength = -1;
	
	private double mean = 0;
	private double stddev = 1;
	private boolean normed = false;

	
	public TimeSeriesInstance(double[] data, double label, int index, int fullLength) 
	{
	    this.data = data;
	    this.label = label;
	    this.m_index = index;
	    this.m_fullLength = fullLength;
	}
	
	public int length(){
		if(this.data == null)
			return 0;
		return this.data.length;	
	}
	
	public TimeSeriesInstance clone(){
//		TimeSeriesInstance ts = new TimeSeriesInstance();
//		ts.label = this.label;
//		ts.data = new double[length()];
//		ts.data = this.data.clone();
//		return ts;
		return new TimeSeriesInstance(this.data, this.label, this.m_index, this.m_fullLength);
	}
	
	public void setIndex(int index)
	{
		m_index = index;
	}
	
	public int getIndex()
	{
		return m_index;
	}
	
	public void setFullLength(int len)
	{
		m_fullLength = len;
	}
	
	public int getFullLength()
	{
		return m_fullLength;
	}
	
	public double[] getData()
	{
		return data;
	}
	
	public double getLabel()
	{
		return label;
	}

	public TimeSeriesInstance subTS(int from, int len)
	{
		double[] instance = Arrays.copyOfRange(data, from, from + len);
		return new TimeSeriesInstance(instance, label, this.m_index, this.m_fullLength);
	}
	
	public TimeSeriesInstance truncateTo(int length)
	{
		int len = Math.min(length, length());
		double[] instance = Arrays.copyOfRange(data, 0, len);
		return new TimeSeriesInstance(instance, label, this.m_index, this.m_fullLength);
	}
	
	public TimeSeriesInstance truncateTo(double ratio, int minLength)
	{
		int len = (int) Math.round(length() * ratio);
		len = Math.min(len, length());
		len = Math.max(len, minLength);
		double[] instance = Arrays.copyOfRange(data, 0, len);
		return new TimeSeriesInstance(instance, label, m_index, this.m_fullLength);
	}
	
	public TimeSeriesInstance truncateBigTo(int length)
	{
		double[] instance = new double[length];
		System.arraycopy(data, 0, instance, 0, data.length);
		
		for(int i = data.length; i < length; i++)
		{
			instance[i] = 0;
		}
		return new TimeSeriesInstance(instance, label, m_index, this.m_fullLength);
	}
	
	public FeatureNode[] toLinearNode(){
		FeatureNode[] node = new FeatureNode[data.length];
		for(int i = 0; i < data.length; i++){
			node[i] = new FeatureNode(i+1, data[i]);
		}
		return node;
	}
	
	public TimeSeries toWEASEL()
	{
		return new TimeSeries(Arrays.copyOfRange(data, 0, data.length), label);
	}
	
	public TimeSeriesInstance paa(int len) 
	{
        if (len >= data.length) 
            return this.clone();
        
        double[] intervals = new double[len];

        //counters to keep track of progress towards completion of a frame
        //potential for data.length % intervals != 0, therefore non-integer
        //interval length, so weight the boundary data points to effect both 
        //intervals it touches
        int currentFrame = 0;
        double realFrameLength = (double)data.length / len;
        double frameSum = 0.0, currentFrameSize = 0.0, remaining = 0.0;
        
        //PAA conversion
        for (int i = 0; i < data.length; ++i) 
        {
            remaining = realFrameLength - currentFrameSize;
            if (remaining > 1.0) 
            {
                //just use whole data point 
                frameSum += data[i];
                currentFrameSize += 1;
            } else 
            {
                //use some portion of data point as needed
                frameSum += remaining * data[i]; 
                currentFrameSize += remaining;
            }   
            
            if (currentFrameSize == realFrameLength) 
            { //if frame complete
                intervals[currentFrame++] = frameSum / realFrameLength; //store mean

                //before going onto next datapoint, 'use up' any of the current one on the new interval
                //that might not have been used for interval just completed
                frameSum = (1-remaining) * data[i];
                currentFrameSize = (1-remaining);
            }
        }
        
        // i.e. if the last interval didn't write because of double imprecision
        if (currentFrame == len-1) { //if frame complete
            intervals[currentFrame++] = frameSum / realFrameLength;
        }
        
        return new TimeSeriesInstance(intervals, label, m_index, m_fullLength);
    }
	
	public boolean isNormed() 
	{
	    return this.normed;
	}
	
	public void norm(boolean normMean) {
	    this.mean = calculateMean();
	    this.stddev = calculateStddev();

	    if (!isNormed()) {
	      norm(normMean, this.mean, this.stddev);
	    }
	  }

	  /**
	   * Used for zero-mean normalization.
	   * @param normMean defines, if the mean should be subtracted from the time series
	   * @param mean the mean to set (usually set to 0)
	   * @param stddev the stddev to set (usually set to 1)
	   */
	  public void norm(boolean normMean, double mean, double stddev) {
	    this.mean = mean;
	    this.stddev = stddev;

	    if (!isNormed()) {
	      double inverseStddev = (this.stddev != 0) ? 1.0 / this.stddev : 1.0;

	      if (normMean) {
	        for (int i = 0; i < this.data.length; i++) {
	          this.data[i] = (this.data[i] - this.mean) * inverseStddev;
	        }
	        this.mean = 0.0;
	      } else if (inverseStddev != 1.0) {
	        for (int i = 0; i < this.data.length; i++) {
	          this.data[i] *= inverseStddev;
	        }
	      }

	      //      this.mean = 0.0;
	      //      this.stddev = 1.0;
	      this.normed = true;
	    }
	  }

	  public double calculateStddev() {
	    this.stddev = 0;

	    // stddev
	    double var = 0;
	    for (double value : getData()) {
	      var += value * value;
	    }

	    double norm = 1.0 / ((double) this.data.length);
	    double buf = norm * var - this.mean * this.mean;
	    if (buf > 0) {
	      this.stddev = Math.sqrt(buf);
	    }

	    return this.stddev;
	  }


	  public double calculateMean() {
	    this.mean = 0.0;

	    // get mean values
	    for (double value : getData()) {
	      this.mean += value;
	    }
	    this.mean /= (double) this.data.length;

	    return this.mean;
	  }
}
