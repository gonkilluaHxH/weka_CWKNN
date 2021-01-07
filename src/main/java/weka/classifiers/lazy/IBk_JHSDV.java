/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 *    IBk_JHSDV.java
 *    Copyright (C) 1999-2012 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.lazy;

import java.util.Collections;
import java.util.Enumeration;
import java.util.Vector;
import java.lang.Math;
import java.util.Arrays;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.UpdateableClassifier;
import weka.classifiers.rules.ZeroR;
import weka.core.AdditionalMeasureProducer;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.RevisionUtils;
import weka.core.SelectedTag;
import weka.core.Tag;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;
import weka.core.neighboursearch.LinearNNSearch;
import weka.core.neighboursearch.NearestNeighbourSearch;
import weka.core.DenseInstance;
/*
 * This class works as follow:
 * start method is the iniater of the class
 * $ define method is for variable declaration it used for having a dynamic sized variables. 
 * after loading the dataset from the constructer define method sets the size of the variables according 
 * to the dataset used.
 * $ the next method initiated in start "cal_sum_and_count" is responsible for calcualating the summation of
 * each attribute per class and the number of occurence of each class.
 * $ the next method is responsible for calculating the centroid of each class and it adds the centroids to the dataset.
 * $ the next method is responsible for filling an 2D array with the centroids. (used to simplify work ahead.).
 * $ the next method calculates the standard deviation. 
 * 
 */
/**
 * <!-- globalinfo-start --> K-nearest neighbours classifier. Can select
 * appropriate value of K based on cross-validation. Can also do distance
 * weighting.<br/>
 * <br/>
 * For more information, see<br/>
 * <br/>
 * D. Aha, D. Kibler (1991). Instance-based learning algorithms. Machine
 * Learning. 6:37-66.
 * <p/>
 * <!-- globalinfo-end -->
 * 
 * <!-- technical-bibtex-start --> BibTeX:
 * 
 * <pre>
 * &#64;article{Aha1991,
 *    author = {D. Aha and D. Kibler},
 *    journal = {Machine Learning},
 *    pages = {37-66},
 *    title = {Instance-based learning algorithms},
 *    volume = {6},
 *    year = {1991}
 * }
 * </pre>
 * <p/>
 * <!-- technical-bibtex-end -->
 *
 * <!-- options-start --> Valid options are:
 * <p/>
 * 
 * <pre>
 *  -I
 *  Weight neighbours by the inverse of their distance
 *  (use when k &gt; 1)
 * </pre>
 * 
 * <pre>
 *  -F
 *  Weight neighbours by 1 - their distance
 *  (use when k &gt; 1)
 * </pre>
 * 
 * <pre>
 *  -K &lt;number of neighbors&gt;
 *  Number of nearest neighbours (k) used in classification.
 *  (Default = 1)
 * </pre>
 * 
 * <pre>
 *  -E
 *  Minimise mean squared error rather than mean absolute
 *  error when using -X option with numeric prediction.
 * </pre>
 * 
 * <pre>
 *  -W &lt;window size&gt;
 *  Maximum number of training instances maintained.
 *  Training instances are dropped FIFO. (Default = no window)
 * </pre>
 * 
 * <pre>
 *  -X
 *  Select the number of nearest neighbours between 1
 *  and the k value specified using hold-one-out evaluation
 *  on the training data (use when k &gt; 1)
 * </pre>
 * 
 * <pre>
 *  -A
 *  The nearest neighbour search algorithm to use (default: weka.core.neighboursearch.LinearNNSearch).
 * </pre>
 * 
 * <!-- options-end -->
 *
 * @author Stuart Inglis (singlis@cs.waikato.ac.nz)
 * @author Len Trigg (trigg@cs.waikato.ac.nz)
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @version $Revision: 10141 $
 */
public class IBk_JHSDV extends AbstractClassifier implements OptionHandler, UpdateableClassifier,
		WeightedInstancesHandler, TechnicalInformationHandler, AdditionalMeasureProducer {

	/** for serialization. */
	static final long serialVersionUID = -3080186098777067172L;

	/** The training instances used for classification. */
	protected static Instances m_Train;

	/** The number of class values (or 1 if predicting numeric). */
	protected static int m_NumClasses;

	/** The class attribute type. */
	protected int m_ClassType;

	/** The number of neighbours to use for classification (currently). */
	protected static int m_kNN;

	/**
	 * The value of kNN provided by the user. This may differ from m_kNN if
	 * cross-validation is being used.
	 */
	protected int m_kNNUpper;

	/**
	 * Whether the value of k selected by cross validation has been invalidated by a
	 * change in the training instances.
	 */
	protected boolean m_kNNValid;

	/**
	 * The maximum number of training instances allowed. When this limit is reached,
	 * old training instances are removed, so the training data is "windowed". Set
	 * to 0 for unlimited numbers of instances.
	 */
	protected int m_WindowSize;

	/** Whether the neighbours should be distance-weighted. */
	protected static int m_DistanceWeighting;

	/** Whether to select k by cross validation. */
	protected boolean m_CrossValidate;

	/**
	 * Whether to minimise mean squared error rather than mean absolute error when
	 * cross-validating on numeric prediction tasks.
	 */
	protected boolean m_MeanSquared;

	/** Default ZeroR model to use when there are no training instances */
	protected ZeroR m_defaultModel;

	/** no weighting. */
	public static final int WEIGHT_NONE = 1;
	/** weight by 1/distance. */
	public static final int WEIGHT_INVERSE = 2;
	/** weight by 1-distance. */
	public static final int WEIGHT_SIMILARITY = 4;

	public static final int WEIGHT_WKS = 5;

	public static final int WEIGHT_SD = 6;

	public static final int WEIGHT_BOTH = 7;
	/** possible instance weighting methods. */

	public static final Tag[] TAGS_WEIGHTING = { new Tag(WEIGHT_NONE, "No distance weighting"),
			new Tag(WEIGHT_INVERSE, "Weight by 1/distance"), new Tag(WEIGHT_SIMILARITY, "Weight by 1-distance"),
			new Tag(WEIGHT_WKS, "Weight by WKS"), new Tag(WEIGHT_SD, "Weight by Sd"),
			new Tag(WEIGHT_BOTH, "Weight by Standard deviation and WKS") };

	/** for nearest-neighbor search. */
	protected NearestNeighbourSearch m_NNSearch = new LinearNNSearch();

	/** The number of attributes the contribute to a prediction. */
	protected double m_NumAttributesUsed;

	protected static Instances temp;

	protected static int number_of_attributes;

	protected static double[][] myArr;

	protected static double[][] centroids_double;

	protected static double[] countClass;

	protected static double[][] SDPerClass;

	protected static double[][] combinations;

	protected static double[] radius;

	protected static double[] weight_class;

	protected static double[] weights_of_wks;

	protected static int classPosition;

	/**
	 * IBk_JHSDV classifier. Simple instance-based learner that uses the class of the
	 * nearest k training instances for the class of the test instances.
	 *
	 * @param k
	 *            the number of nearest neighbors to use for prediction
	 */

	public IBk_JHSDV(int k) {

		init();
		setKNN(k);
	}

	public IBk_JHSDV(Instances trainSet, int kvalue) throws Exception { // my constructer
		m_Train = trainSet;
		temp = trainSet;
		number_of_attributes = trainSet.numAttributes();
		m_NumClasses = trainSet.numClasses();
		classPosition = trainSet.numAttributes() - 1;
		m_kNN = kvalue;
		start();

	}

	protected static void start() throws Exception {
		System.out.println("start().....");
		define(); // for variable declaration.
		WKS_WEIGHT(); //calculate the weight of each instance accoring to wks method.
		calSumAndCount(m_Train);//calculate the summation of each attribute and the occurence of each class.
		Instances all = calCentroids(); //calcualte the centroids.
		centroid_double(); // creates an array of centroids
		cal_standard_deviaion_array(); //calcualte standard deviation.
		calculate_radius(); // calculate the redias of accaptable deviation
		array_of_combination();
		operate();
		ATTWeight();//sets the weight of the each instance.
	}

	protected static void define() {
		int permitation = number_of_combination();

		myArr = new double[m_NumClasses][number_of_attributes];

		countClass = new double[m_NumClasses];

		centroids_double = new double[m_NumClasses][number_of_attributes];

		SDPerClass = new double[m_NumClasses][number_of_attributes];

		radius = new double[m_NumClasses];

		combinations = new double[permitation / 2][2];

		weight_class = new double[m_NumClasses];

		weights_of_wks = new double[m_Train.numInstances()];

	}

	/*
	 * in order to calculate the distance between classes using a custimized
	 * distance method i have tow options option one is to add the new distance
	 * method in the distance class yet this method needs some value computed in
	 * this class this needs comunication between tow classes and takes more time to
	 * process the data. secend method is to add a distance method in this class.
	 * this option have one problem picking the classes combination to compute the
	 * distance between all the classes.in order to get the combination I used
	 * probability to calculate the number of combination of all classes. then I
	 * created an array of the size classes this array contain numbers from zero to
	 * number of classes. this array is represents the index of centroids of
	 * instances. used indexing in order to simplify the work on users and the
	 * processer. step one is to calculate the factorial step tow is to apply
	 * combination permitation rule. step three is to fill a string with all the
	 * combinations. step four is to fill tow dimentional array with the centroids
	 * combinations.
	 */
	// calculate the factorial value.
	protected static int factorial(int n) {
		if (n == 0)
			return 1;
		else
			return (n * factorial(n - 1));
	}

	// calculate the size of the array of combination of centroids
	protected static int number_of_combination() {
		int result = factorial(m_NumClasses);
		int size = result / (2 * (factorial((m_NumClasses - 2))));
		return size;
	}

	// this method calculate the summation of each attribute per class.
	public static double[][] calSumAndCount(Instances data) {
		data = m_Train;
		data.setClassIndex(classPosition);

		for (int i = 0; i < data.numInstances(); i++) {
			Instance nn = data.get(i);
			int n = (int) nn.classValue();

			countClass[(int) n]++;

			myArr[(int) n][(int) data.numAttributes() - 1] += 1;

			for (int x = 0; x < nn.numAttributes() - 1; x++) {
				myArr[(int) n][x] += nn.value(x);
			}
		}
		return myArr;
	}

	// this method calculates the centroids of the data.
	public static Instances calCentroids() {
		Instance centerInstance = new DenseInstance(temp.numAttributes());
		centerInstance.setDataset(temp);
		for (int j = 0; j < myArr.length; j++) {

			for (int k = 0; k < myArr[j].length - 1; k++) {

				centerInstance.setValue(k, myArr[j][k] / myArr[j][temp.numAttributes() - 1]);
				centerInstance.setClassValue((double) j);
			}
			centerInstance.setValue(temp.numAttributes() - 1, j);
			temp.add(centerInstance);
		}

		return temp;
	}

	// this method retrives the centriods of the data per class
	protected static double[][] centroid_double() {
		for (int f = 0; f < centroids_double.length; f++) {
			for (int d = 0; d < centroids_double[0].length; d++) {
				centroids_double[f][d] = temp.get(temp.numInstances() - temp.numClasses() + f).value(d);
			}
		}
		return centroids_double;
	}

	protected static double[][] cal_standard_deviaion_array() {
		double total;

		for (int attribute_position = 0; attribute_position < number_of_attributes - 1; attribute_position++) {

			total = 0;
			for (int centroidscount = 0; centroidscount < centroids_double.length; centroidscount++) { // of centroids
				total = 0;
				for (int i = 0; i < temp.numInstances(); i++) { // of instance
					Instance instance = temp.get(i);
					if (instance.value(
							temp.numAttributes() - 1) == centroids_double[centroidscount][temp.numAttributes() - 1]) { // check
						double test = instance.value(attribute_position)
								- centroids_double[centroidscount][attribute_position];
						total += Math.pow(test, 2);
						continue;
					}
				}

				total = total / myArr[centroidscount][temp.numAttributes() - 1];
				total = Math.sqrt(total);

				SDPerClass[centroidscount][attribute_position] = total;
				SDPerClass[centroidscount][temp.numAttributes() - 1] = centroidscount;

			}
		}
		return SDPerClass;
	}

	// this method takes the string of combinations and returns tow dimensional
	// array each array containing tow index of centroids of the data.
	protected static double[][] array_of_combination() {
		String n = posibilities();
		String[] number = n.split(",");
		int[] possibilities_array = new int[number.length];
		for (int i = 0; i < number.length; i++) {
			possibilities_array[i] = Integer.parseInt(number[i]);
		}
		double[][] posibile_array = null;
		posibile_array = new double[possibilities_array.length / 2][2];
		int count = 0;
		for (int i = 0; i < possibilities_array.length / 2; i++) {
			posibile_array[i][0] = possibilities_array[i + count];
			posibile_array[i][1] = possibilities_array[i + 1 + count];
			count += 1;
		}
		combinations = posibile_array;
		return posibile_array;
	}

	// this method returns a string containing the combinations separated with ","
	protected static String posibilities() {
		String S = new String();
		int number_of_possibilities = number_of_combination() * 4;
		for (int i = 0; i < m_NumClasses; i++) {
			for (int j = 1; j < m_NumClasses; j++) {
				if (i == 1 & j == 1) {
					j += 1;
				} else if (i > 1 && j == 1) {
					j += i;
				}
				if (S.length() < number_of_possibilities)
					S += i + "," + j + ",";
			}
		}
		return S;
	}

	// this method calcutes the radius of all classes. radius is equal to the
	// summmation of standard deviation of all att of each class
	protected static double[] calculate_radius() {
		for (int i = 0; i < temp.numClasses(); i++) {
			double radias = 0;
			for (int j = 0; j < number_of_attributes - 1; j++) {
				radias += Math.pow(SDPerClass[i][j], 2);
			}
			radias = Math.sqrt(radias);
			radius[i] = radias;
		}
		System.out.println("in the calculate radius methods" + Arrays.toString(radius));
		return radius;

	}

	// this method calculates intersections and creates an array that represents the
	// number of intersections.
	protected static double[] operate() {
		int[] result_ofmanhaboulice = new int[centroids_double.length];
		double[] intersections = new double[centroids_double.length];

		for (int combination_position = 0; combination_position < combinations.length; combination_position++) { // e7temal
			int _pos1 = (int) combinations[combination_position][0];
			int _pos2 = (int) combinations[combination_position][1];
			double total = 0;
			double radius_sum = 0;
			double _radius1 = radius[_pos1];
			double _radius2 = radius[_pos2];
			for (int nbofatt = 0; nbofatt < centroids_double[0].length - 1; nbofatt++) {

				double cent_att1 = centroids_double[_pos1][nbofatt];
				double cent_att2 = centroids_double[_pos2][nbofatt];
				double stand_att1 = SDPerClass[_pos1][nbofatt];
				double stand_att2 = SDPerClass[_pos2][nbofatt];
				double cent_distance_per_attribute = Math.pow((cent_att1 - cent_att2), 2);
				double standard_deviation_mean_of_each_att = (stand_att1 + stand_att2) / 2;
				double distance_per_attribute = cent_distance_per_attribute / standard_deviation_mean_of_each_att;
				total += distance_per_attribute;

			}
			total = Math.sqrt(total);
			radius_sum = _radius1 + _radius2;
			if (total >= radius_sum) {
				System.out.println("distance is larger than the sum of rediases.");
				result_ofmanhaboulice[_pos1] += 1 / m_NumClasses;
				result_ofmanhaboulice[_pos2] += 1 / m_NumClasses;
			}

		}

		for (int i = 0; i < result_ofmanhaboulice.length; i++) {

			double result_of_manhabules_number = result_ofmanhaboulice[i];

			result_of_manhabules_number = result_of_manhabules_number - 1;

			result_of_manhabules_number = result_of_manhabules_number / result_ofmanhaboulice.length;

			result_of_manhabules_number = result_of_manhabules_number + 1;

			result_of_manhabules_number = Math.exp(-result_of_manhabules_number);

			result_of_manhabules_number = 1 + result_of_manhabules_number;

			result_of_manhabules_number = 1 / result_of_manhabules_number;

			intersections[i] = result_of_manhabules_number;

		}
		weight_class = intersections;
		System.out.println("array  of weights based on intersections  per class " + Arrays.toString(intersections));
		return intersections;

	}

	public static void WKS_WEIGHT() throws Exception {
		Instance n = temp.get(0);
		for (int _pos = 0; _pos < temp.numInstances() - temp.numClasses(); _pos++) {
			double right_wrong = 0;
			double final_result = 0;

			LinearNNSearch search = new LinearNNSearch(m_Train);
			// select the instance to start
			// gets the nearest neighbors of the selected instance
			Instances nearest = search.kNearestNeighbours(n, m_kNN); // 5 is the K (number of nearest neighbors)//here

			double right = 0;
			double wrong = 0;
			// calculate weight of the selected instance
			for (int j = 0; j < nearest.numInstances(); j++) {

				// checks if the class of the selected instance equals to the class of the
				// neighbors
				if (n.classAttribute().value((int) n.classValue()).equalsIgnoreCase(
						nearest.instance(j).classAttribute().value((int) nearest.instance(j).classValue()))) {
					right++;
				} else {
					wrong++;
				}
			}
			right_wrong = (right - wrong) / (right + wrong);
			final_result = 1 / (1 + (Math.exp(-right_wrong)));

			weights_of_wks[_pos] = final_result;

		}

	}

	// this method sets the weight of all the data instances.

	public static Instances ATTWeight() throws Exception {
		for (int i = 0; i < temp.numInstances() - temp.numClasses(); i++) {
			Instance n = temp.get(i);
			for (int j = 0; j < weight_class.length; j++) {
				if ((int) n.classValue() == j) {
					// double weight_of_instance= weights_of_wks [j] + weight_class[j];
					double weight_of_instance = weight_class[j];
					n.setWeight(weight_of_instance);
				}
			}

		}
		return temp;
	}

	protected double instanceweight(Instance instance) {
		double wieght_of_instance = instance.weight();
		return wieght_of_instance;
	}

	/**
	 * IB1 classifer. Instance-based learner. Predicts the class of the single
	 * nearest training instance for each test instance.
	 */
	public IBk_JHSDV() {

		init();
	}

	/**
	 * Returns a string describing classifier.
	 * 
	 * @return a description suitable for displaying in the explorer/experimenter
	 *         gui
	 */
	public String globalInfo() {

		return "K-nearest neighbours classifier. Can "
				+ "select appropriate value of K based on cross-validation. Can also do " + "distance weighting.\n\n"
				+ "For more information, see\n\n" + getTechnicalInformation().toString();
	}

	/**
	 * Returns an instance of a TechnicalInformation object, containing detailed
	 * information about the technical background of this class, e.g., paper
	 * reference or book this class is based on.
	 * 
	 * @return the technical information about this class
	 */
	public TechnicalInformation getTechnicalInformation() {
		TechnicalInformation result;

		result = new TechnicalInformation(Type.ARTICLE);
		result.setValue(Field.AUTHOR, "D. Aha and D. Kibler");
		result.setValue(Field.YEAR, "1991");
		result.setValue(Field.TITLE, "Instance-based learning algorithms");
		result.setValue(Field.JOURNAL, "Machine Learning");
		result.setValue(Field.VOLUME, "6");
		result.setValue(Field.PAGES, "37-66");

		return result;
	}

	/**
	 * Returns the tip text for this property.
	 * 
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String KNNTipText() {
		return "The number of neighbours to use.";
	}

	/**
	 * Set the number of neighbours the learner is to use.
	 *
	 * @param k
	 *            the number of neighbours.
	 */
	public void setKNN(int k) {
		m_kNN = k;
		m_kNNUpper = k;
		m_kNNValid = false;
	}

	/**
	 * Gets the number of neighbours the learner will use.
	 *
	 * @return the number of neighbours.
	 */
	public int getKNN() {

		return m_kNN;
	}

	/**
	 * Returns the tip text for this property.
	 * 
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String windowSizeTipText() {
		return "Gets the maximum number of instances allowed in the training "
				+ "pool. The addition of new instances above this value will result "
				+ "in old instances being removed. A value of 0 signifies no limit "
				+ "to the number of training instances.";
	}

	/**
	 * Gets the maximum number of instances allowed in the training pool. The
	 * addition of new instances above this value will result in old instances being
	 * removed. A value of 0 signifies no limit to the number of training instances.
	 *
	 * @return Value of WindowSize.
	 */
	public int getWindowSize() {

		return m_WindowSize;
	}

	/**
	 * Sets the maximum number of instances allowed in the training pool. The
	 * addition of new instances above this value will result in old instances being
	 * removed. A value of 0 signifies no limit to the number of training instances.
	 *
	 * @param newWindowSize
	 *            Value to assign to WindowSize.
	 */
	public void setWindowSize(int newWindowSize) {

		m_WindowSize = newWindowSize;
	}

	/**
	 * Returns the tip text for this property.
	 * 
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String distanceWeightingTipText() {

		return "Gets the distance weighting method used.";
	}

	/**
	 * Gets the distance weighting method used. Will be one of WEIGHT_NONE,
	 * WEIGHT_INVERSE, or WEIGHT_SIMILARITY
	 *
	 * @return the distance weighting method used.
	 */
	public static SelectedTag getDistanceWeighting() throws Exception {
		return new SelectedTag(m_DistanceWeighting, TAGS_WEIGHTING);

	}

	/**
	 * Sets the distance weighting method used. Values other than WEIGHT_NONE,
	 * WEIGHT_INVERSE, or WEIGHT_SIMILARITY will be ignored.
	 *
	 * @param newMethod
	 *            the distance weighting method to use
	 */
	public static void setDistanceWeighting(SelectedTag newMethod) {

		if (newMethod.getTags() == TAGS_WEIGHTING) {
			m_DistanceWeighting = newMethod.getSelectedTag().getID();
		}
	}

	/**
	 * Returns the tip text for this property.
	 * 
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String meanSquaredTipText() {

		return "Whether the mean squared error is used rather than mean "
				+ "absolute error when doing cross-validation for regression problems.";
	}

	/**
	 * Gets whether the mean squared error is used rather than mean absolute error
	 * when doing cross-validation.
	 *
	 * @return true if so.
	 */
	public boolean getMeanSquared() {

		return m_MeanSquared;
	}

	/**
	 * Sets whether the mean squared error is used rather than mean absolute error
	 * when doing cross-validation.
	 *
	 * @param newMeanSquared
	 *            true if so.
	 */
	public void setMeanSquared(boolean newMeanSquared) {

		m_MeanSquared = newMeanSquared;
	}

	/**
	 * Returns the tip text for this property.
	 * 
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String crossValidateTipText() {

		return "Whether hold-one-out cross-validation will be used to "
				+ "select the best k value between 1 and the value specified as " + "the KNN parameter.";
	}

	/**
	 * Gets whether hold-one-out cross-validation will be used to select the best k
	 * value.
	 *
	 * @return true if cross-validation will be used.
	 */
	public boolean getCrossValidate() {

		return m_CrossValidate;
	}

	/**
	 * Sets whether hold-one-out cross-validation will be used to select the best k
	 * value.
	 *
	 * @param newCrossValidate
	 *            true if cross-validation should be used.
	 */
	public void setCrossValidate(boolean newCrossValidate) {

		m_CrossValidate = newCrossValidate;
	}

	/**
	 * Returns the tip text for this property.
	 * 
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String nearestNeighbourSearchAlgorithmTipText() {
		return "The nearest neighbour search algorithm to use "
				+ "(Default: weka.core.neighboursearch.LinearNNSearch).";
	}

	/**
	 * Returns the current nearestNeighbourSearch algorithm in use.
	 * 
	 * @return the NearestNeighbourSearch algorithm currently in use.
	 */
	public NearestNeighbourSearch getNearestNeighbourSearchAlgorithm() {
		return m_NNSearch;
	}

	/**
	 * Sets the nearestNeighbourSearch algorithm to be used for finding nearest
	 * neighbour(s).
	 * 
	 * @param nearestNeighbourSearchAlgorithm
	 *            - The NearestNeighbourSearch class.
	 */
	public void setNearestNeighbourSearchAlgorithm(NearestNeighbourSearch nearestNeighbourSearchAlgorithm) {
		m_NNSearch = nearestNeighbourSearchAlgorithm;
	}

	/**
	 * Get the number of training instances the classifier is currently using.
	 * 
	 * @return the number of training instances the classifier is currently using
	 */
	public int getNumTraining() {

		return m_Train.numInstances();
	}

	/**
	 * Returns default capabilities of the classifier.
	 *
	 * @return the capabilities of this classifier
	 */
	public Capabilities getCapabilities() {
		Capabilities result = super.getCapabilities();
		result.disableAll();

		// attributes
		result.enable(Capability.NOMINAL_ATTRIBUTES);
		result.enable(Capability.NUMERIC_ATTRIBUTES);
		result.enable(Capability.DATE_ATTRIBUTES);
		result.enable(Capability.MISSING_VALUES);

		// class
		result.enable(Capability.NOMINAL_CLASS);
		result.enable(Capability.NUMERIC_CLASS);
		result.enable(Capability.DATE_CLASS);
		result.enable(Capability.MISSING_CLASS_VALUES);

		// instances
		result.setMinimumNumberInstances(0);

		return result;
	}

	/**
	 * Generates the classifier.
	 *
	 * @param instances
	 *            set of instances serving as training data
	 * @throws Exception
	 *             if the classifier has not been generated successfully
	 */
	public void buildClassifier(Instances instances) throws Exception {

		// can classifier handle the data?
		getCapabilities().testWithFail(instances);

		// remove instances with missing class
		instances = new Instances(instances);
		instances.deleteWithMissingClass();

		m_NumClasses = instances.numClasses();
		m_ClassType = instances.classAttribute().type();
		m_Train = new Instances(instances, 0, instances.numInstances());

		// Throw away initial instances until within the specified window size
		if ((m_WindowSize > 0) && (instances.numInstances() > m_WindowSize)) {
			m_Train = new Instances(m_Train, m_Train.numInstances() - m_WindowSize, m_WindowSize);
		}

		m_NumAttributesUsed = 0.0;
		for (int i = 0; i < m_Train.numAttributes(); i++) {
			if ((i != m_Train.classIndex()) && (m_Train.attribute(i).isNominal() || m_Train.attribute(i).isNumeric())) {
				m_NumAttributesUsed += 1.0;
			}
		}

		m_NNSearch.setInstances(m_Train);

		// Invalidate any currently cross-validation selected k
		m_kNNValid = false;

		m_defaultModel = new ZeroR();
		m_defaultModel.buildClassifier(instances);
	}

	/**
	 * Adds the supplied instance to the training set.
	 *
	 * @param instance
	 *            the instance to add
	 * @throws Exception
	 *             if instance could not be incorporated successfully
	 */
	public void updateClassifier(Instance instance) throws Exception {

		if (m_Train.equalHeaders(instance.dataset()) == false) {
			throw new Exception("Incompatible instance types\n" + m_Train.equalHeadersMsg(instance.dataset()));
		}
		if (instance.classIsMissing()) {
			return;
		}

		m_Train.add(instance);
		m_NNSearch.update(instance);
		m_kNNValid = false;
		if ((m_WindowSize > 0) && (m_Train.numInstances() > m_WindowSize)) {
			boolean deletedInstance = false;
			while (m_Train.numInstances() > m_WindowSize) {
				m_Train.delete(0);
				deletedInstance = true;
			}
			// rebuild datastructure KDTree currently can't delete
			if (deletedInstance == true)
				m_NNSearch.setInstances(m_Train);
		}
	}

	/**
	 * Calculates the class membership probabilities for the given test instance.
	 *
	 * @param instance
	 *            the instance to be classified
	 * @return predicted class probability distribution
	 * @throws Exception
	 *             if an error occurred during the prediction
	 */
	public double[] distributionForInstance(Instance instance) throws Exception {

		if (m_Train.numInstances() == 0) {
			// throw new Exception("No training instances!");
			return m_defaultModel.distributionForInstance(instance);
		}
		if ((m_WindowSize > 0) && (m_Train.numInstances() > m_WindowSize)) {
			m_kNNValid = false;
			boolean deletedInstance = false;
			while (m_Train.numInstances() > m_WindowSize) {
				m_Train.delete(0);
			}
			// rebuild datastructure KDTree currently can't delete
			if (deletedInstance == true)
				m_NNSearch.setInstances(m_Train);
		}

		// Select k by cross validation
		if (!m_kNNValid && (m_CrossValidate) && (m_kNNUpper >= 1)) {
			crossValidate();
		}

		m_NNSearch.addInstanceInfo(instance);

		Instances neighbours = m_NNSearch.kNearestNeighbours(instance, m_kNN);
		double[] distances = m_NNSearch.getDistances();
		double[] distribution = makeDistribution(neighbours, distances);

		return distribution;
	}

	/**
	 * Returns an enumeration describing the available options.
	 *
	 * @return an enumeration of all the available options.
	 */
	public Enumeration<Option> listOptions() {

		Vector<Option> newVector = new Vector<Option>(7);

		newVector.addElement(new Option("\tWeight neighbours by the inverse of their distance\n" + "\t(use when k > 1)",
				"I", 0, "-I"));
		newVector.addElement(
				new Option("\tWeight neighbours by 1 - their distance\n" + "\t(use when k > 1)", "F", 0, "-F"));
		newVector.addElement(
				new Option("\tNumber of nearest neighbours (k) used in classification.\n" + "\t(Default = 1)", "K", 1,
						"-K <number of neighbors>"));
		newVector.addElement(new Option("\tMinimise mean squared error rather than mean absolute\n"
				+ "\terror when using -X option with numeric prediction.", "E", 0, "-E"));
		newVector.addElement(new Option("\tMaximum number of training instances maintained.\n"
				+ "\tTraining instances are dropped FIFO. (Default = no window)", "W", 1, "-W <window size>"));
		newVector.addElement(new Option("\tSelect the number of nearest neighbours between 1\n"
				+ "\tand the k value specified using hold-one-out evaluation\n"
				+ "\ton the training data (use when k > 1)", "X", 0, "-X"));
		newVector.addElement(new Option("\tThe nearest neighbour search algorithm to use "
				+ "(default: weka.core.neighboursearch.LinearNNSearch).\n", "A", 0, "-A"));

		newVector.addAll(Collections.list(super.listOptions()));

		return newVector.elements();
	}

	/**
	 * Parses a given list of options.
	 * <p/>
	 *
	 * <!-- options-start --> Valid options are:
	 * <p/>
	 * 
	 * <pre>
	 *  -I
	 *  Weight neighbours by the inverse of their distance
	 *  (use when k &gt; 1)
	 * </pre>
	 * 
	 * <pre>
	 *  -F
	 *  Weight neighbours by 1 - their distance
	 *  (use when k &gt; 1)
	 * </pre>
	 * 
	 * <pre>
	 *  -K &lt;number of neighbors&gt;
	 *  Number of nearest neighbours (k) used in classification.
	 *  (Default = 1)
	 * </pre>
	 * 
	 * <pre>
	 *  -E
	 *  Minimise mean squared error rather than mean absolute
	 *  error when using -X option with numeric prediction.
	 * </pre>
	 * 
	 * <pre>
	 *  -W &lt;window size&gt;
	 *  Maximum number of training instances maintained.
	 *  Training instances are dropped FIFO. (Default = no window)
	 * </pre>
	 * 
	 * <pre>
	 *  -X
	 *  Select the number of nearest neighbours between 1
	 *  and the k value specified using hold-one-out evaluation
	 *  on the training data (use when k &gt; 1)
	 * </pre>
	 * 
	 * <pre>
	 *  -A
	 *  The nearest neighbour search algorithm to use (default: weka.core.neighboursearch.LinearNNSearch).
	 * </pre>
	 * 
	 * <!-- options-end -->
	 *
	 * @param options
	 *            the list of options as an array of strings
	 * @throws Exception
	 *             if an option is not supported
	 */
	public void setOptions(String[] options) throws Exception {

		String knnString = Utils.getOption('K', options);
		if (knnString.length() != 0) {
			setKNN(Integer.parseInt(knnString));
		} else {
			setKNN(1);
		}
		String windowString = Utils.getOption('W', options);
		if (windowString.length() != 0) {
			setWindowSize(Integer.parseInt(windowString));
		} else {
			setWindowSize(0);
		}
		if (Utils.getFlag('I', options)) {
			setDistanceWeighting(new SelectedTag(WEIGHT_INVERSE, TAGS_WEIGHTING));
		} else if (Utils.getFlag('F', options)) {
			setDistanceWeighting(new SelectedTag(WEIGHT_SIMILARITY, TAGS_WEIGHTING));
		} else if (Utils.getFlag('R', options)) {
			setDistanceWeighting(new SelectedTag(WEIGHT_WKS, TAGS_WEIGHTING));
		} else if (Utils.getFlag('S', options)) {
			setDistanceWeighting(new SelectedTag(WEIGHT_SD, TAGS_WEIGHTING));
		} else if (Utils.getFlag('B', options)) {
			setDistanceWeighting(new SelectedTag(WEIGHT_BOTH, TAGS_WEIGHTING));
		} else {
			setDistanceWeighting(new SelectedTag(WEIGHT_NONE, TAGS_WEIGHTING));
		}
		setCrossValidate(Utils.getFlag('X', options));
		setMeanSquared(Utils.getFlag('E', options));

		String nnSearchClass = Utils.getOption('A', options);
		if (nnSearchClass.length() != 0) {
			String nnSearchClassSpec[] = Utils.splitOptions(nnSearchClass);
			if (nnSearchClassSpec.length == 0) {
				throw new Exception("Invalid NearestNeighbourSearch algorithm " + "specification string.");
			}
			String className = nnSearchClassSpec[0];
			nnSearchClassSpec[0] = "";

			setNearestNeighbourSearchAlgorithm(
					(NearestNeighbourSearch) Utils.forName(NearestNeighbourSearch.class, className, nnSearchClassSpec));
		} else
			this.setNearestNeighbourSearchAlgorithm(new LinearNNSearch());

		super.setOptions(options);

		Utils.checkForRemainingOptions(options);
	}

	/**
	 * Gets the current settings of IBk_JHSDV.
	 *
	 * @return an array of strings suitable for passing to setOptions()
	 */
	public String[] getOptions() {

		Vector<String> options = new Vector<String>();
		options.add("-K");
		options.add("" + getKNN());
		options.add("-W");
		options.add("" + m_WindowSize);
		if (getCrossValidate()) {
			options.add("-X");
		}
		if (getMeanSquared()) {
			options.add("-E");
		}
		if (m_DistanceWeighting == WEIGHT_INVERSE) {
			options.add("-I");
		} else if (m_DistanceWeighting == WEIGHT_SIMILARITY) {
			options.add("-F");
		} else if (m_DistanceWeighting == WEIGHT_WKS) {
			options.add("-R");
		} else if (m_DistanceWeighting == WEIGHT_SD) {
			options.add("-S");
		} else if (m_DistanceWeighting == WEIGHT_BOTH) {
			options.add("-B");
		}

		options.add("-A");
		options.add(m_NNSearch.getClass().getName() + " " + Utils.joinOptions(m_NNSearch.getOptions()));

		Collections.addAll(options, super.getOptions());

		return options.toArray(new String[0]);
	}

	/**
	 * Returns an enumeration of the additional measure names produced by the
	 * neighbour search algorithm, plus the chosen K in case cross-validation is
	 * enabled.
	 * 
	 * @return an enumeration of the measure names
	 */
	public Enumeration<String> enumerateMeasures() {
		if (m_CrossValidate) {
			Enumeration<String> enm = m_NNSearch.enumerateMeasures();
			Vector<String> measures = new Vector<String>();
			while (enm.hasMoreElements())
				measures.add(enm.nextElement());
			measures.add("measureKNN");
			return measures.elements();
		} else {
			return m_NNSearch.enumerateMeasures();
		}
	}

	/**
	 * Returns the value of the named measure from the neighbour search algorithm,
	 * plus the chosen K in case cross-validation is enabled.
	 * 
	 * @param additionalMeasureName
	 *            the name of the measure to query for its value
	 * @return the value of the named measure
	 * @throws IllegalArgumentException
	 *             if the named measure is not supported
	 */
	public double getMeasure(String additionalMeasureName) {
		if (additionalMeasureName.equals("measureKNN"))
			return m_kNN;
		else
			return m_NNSearch.getMeasure(additionalMeasureName);
	}

	/**
	 * Returns a description of this classifier.
	 *
	 * @return a description of this classifier as a string.
	 */
	public String toString() {

		if (m_Train == null) {
			return "IBk_JHSDV: No model built yet.";
		}

		if (m_Train.numInstances() == 0) {
			return "Warning: no training instances - ZeroR model used.";
		}

		if (!m_kNNValid && m_CrossValidate) {
			crossValidate();
		}

		String result = "IB1 instance-based classifier\n" + "using " + m_kNN;

		switch (m_DistanceWeighting) {
		case WEIGHT_INVERSE:
			result += " inverse-distance-weighted";
			break;
		case WEIGHT_SIMILARITY:
			result += " similarity-weighted";
			break;
		case WEIGHT_WKS:
			result += " weight_WKS";
			break;
		case WEIGHT_SD:
			result += " weight SD";
			break;
		case WEIGHT_BOTH:
			result += " weight SD and wks";
			break;
		}
		result += " nearest neighbour(s) for classification\n";

		if (m_WindowSize != 0) {
			result += "using a maximum of " + m_WindowSize + " (windowed) training instances\n";
		}
		return result;
	}

	/**
	 * Initialise scheme variables.
	 */
	protected void init() {

		setKNN(1);
		m_WindowSize = 0;
		m_DistanceWeighting = WEIGHT_NONE;
		m_CrossValidate = false;
		m_MeanSquared = false;
	}

	/**
	 * Turn the list of nearest neighbors into a probability distribution.
	 *
	 * @param neighbours
	 *            the list of nearest neighboring instances
	 * @param distances
	 *            the distances of the neighbors
	 * @return the probability distribution
	 * @throws Exception
	 *             if computation goes wrong or has no class attribute
	 */
	protected static int count = 0;

	protected double[] makeDistribution(Instances neighbours, double[] distances) throws Exception {
		count++;
		// System.out.println("make distribution mehtho is running."+ count);
		double total = 0, weight = 0;
		double[] distribution = new double[m_NumClasses];
		// for (int x =0; x < temp.numInstances();x++) {
		// weight = temp.get(x).weight();
		// }
		// Set up a correction to the estimator
		if (m_ClassType == Attribute.NOMINAL) {
			for (int i = 0; i < m_NumClasses; i++) {
				distribution[i] = 1.0 / Math.max(1, m_Train.numInstances());
			}
			total = (double) m_NumClasses / Math.max(1, m_Train.numInstances());
		}

		for (int i = 0; i < neighbours.numInstances(); i++) {
			// Collect class counts
			Instance current = neighbours.instance(i);
			distances[i] = distances[i] * distances[i];
			distances[i] = Math.sqrt(distances[i] / m_NumAttributesUsed);

			switch (m_DistanceWeighting) {
			case WEIGHT_INVERSE:
				weight = 1.0 / (distances[i] + 0.001); // to avoid div by zero
				break;
			case WEIGHT_SIMILARITY:
				weight = 1.0 - distances[i];
				break;
			case WEIGHT_WKS:
				weight = 1;
				break;
			case WEIGHT_SD:
				// current = temp.get(i);
				// weight = current.weight();
				weight = distances[i] / (1 / instanceweight(neighbours.instance(i)));
				break;
			case WEIGHT_BOTH:
				weight = distances[i] / (1 / (((instanceweight(neighbours.instance(i))) / (1/weights_of_wks[i]))));
				break;
			default: // WEIGHT_NONE:
				weight = 1.0;
				break;
			}
			weight *= current.weight();
			try {
				switch (m_ClassType) {
				case Attribute.NOMINAL:
					distribution[(int) current.classValue()] += weight;
					break;
				case Attribute.NUMERIC:
					distribution[0] += current.classValue() * weight;
					break;
				}
			} catch (Exception ex) {
				throw new Error("Data has no class attribute!");
			}
			total += weight;
		}

		// Normalise distribution
		if (total > 0) {
			Utils.normalize(distribution, total);
		}
		return distribution;
	}

	/**
	 * Select the best value for k by hold-one-out cross-validation. If the class
	 * attribute is nominal, classification error is minimised. If the class
	 * attribute is numeric, mean absolute error is minimised
	 */
	protected void crossValidate() {

		try {
			if (m_NNSearch instanceof weka.core.neighboursearch.CoverTree)
				throw new Exception(
						"CoverTree doesn't support hold-one-out " + "cross-validation. Use some other NN " + "method.");

			double[] performanceStats = new double[m_kNNUpper];
			double[] performanceStatsSq = new double[m_kNNUpper];

			for (int i = 0; i < m_kNNUpper; i++) {
				performanceStats[i] = 0;
				performanceStatsSq[i] = 0;
			}

			m_kNN = m_kNNUpper;
			Instance instance;
			Instances neighbours;
			double[] origDistances, convertedDistances;
			for (int i = 0; i < m_Train.numInstances(); i++) {
				if (m_Debug && (i % 50 == 0)) {
					System.err.print("Cross validating " + i + "/" + m_Train.numInstances() + "\r");
				}
				instance = m_Train.instance(i);
				neighbours = m_NNSearch.kNearestNeighbours(instance, m_kNN);
				origDistances = m_NNSearch.getDistances();

				for (int j = m_kNNUpper - 1; j >= 0; j--) {
					// Update the performance stats
					convertedDistances = new double[origDistances.length];
					System.arraycopy(origDistances, 0, convertedDistances, 0, origDistances.length);
					double[] distribution = makeDistribution(neighbours, convertedDistances);
					double thisPrediction = Utils.maxIndex(distribution);
					if (m_Train.classAttribute().isNumeric()) {
						thisPrediction = distribution[0];
						double err = thisPrediction - instance.classValue();
						performanceStatsSq[j] += err * err; // Squared error
						performanceStats[j] += Math.abs(err); // Absolute error
					} else {
						if (thisPrediction != instance.classValue()) {
							performanceStats[j]++; // Classification error
						}
					}
					if (j >= 1) {
						neighbours = pruneToK(neighbours, convertedDistances, j);
					}
				}
			}

			// Display the results of the cross-validation
			for (int i = 0; i < m_kNNUpper; i++) {
				if (m_Debug) {
					System.err.print("Hold-one-out performance of " + (i + 1) + " neighbors ");
				}
				if (m_Train.classAttribute().isNumeric()) {
					if (m_Debug) {
						if (m_MeanSquared) {
							System.err.println("(RMSE) = " + Math.sqrt(performanceStatsSq[i] / m_Train.numInstances()));
						} else {
							System.err.println("(MAE) = " + performanceStats[i] / m_Train.numInstances());
						}
					}
				} else {
					if (m_Debug) {
						System.err.println("(%ERR) = " + 100.0 * performanceStats[i] / m_Train.numInstances());
					}
				}
			}

			// Check through the performance stats and select the best
			// k value (or the lowest k if more than one best)
			double[] searchStats = performanceStats;
			if (m_Train.classAttribute().isNumeric() && m_MeanSquared) {
				searchStats = performanceStatsSq;
			}
			double bestPerformance = Double.NaN;
			int bestK = 1;
			for (int i = 0; i < m_kNNUpper; i++) {
				if (Double.isNaN(bestPerformance) || (bestPerformance > searchStats[i])) {
					bestPerformance = searchStats[i];
					bestK = i + 1;
				}
			}
			m_kNN = bestK;
			if (m_Debug) {
				System.err.println("Selected k = " + bestK);
			}

			m_kNNValid = true;
		} catch (Exception ex) {
			throw new Error("Couldn't optimize by cross-validation: " + ex.getMessage());
		}
	}

	/**
	 * Prunes the list to contain the k nearest neighbors. If there are multiple
	 * neighbors at the k'th distance, all will be kept.
	 *
	 * @param neighbours
	 *            the neighbour instances.
	 * @param distances
	 *            the distances of the neighbours from target instance.
	 * @param k
	 *            the number of neighbors to keep.
	 * @return the pruned neighbours.
	 */
	public Instances pruneToK(Instances neighbours, double[] distances, int k) {

		if (neighbours == null || distances == null || neighbours.numInstances() == 0) {
			return null;
		}
		if (k < 1) {
			k = 1;
		}

		int currentK = 0;
		double currentDist;
		for (int i = 0; i < neighbours.numInstances(); i++) {
			currentK++;
			currentDist = distances[i];
			if (currentK > k && currentDist != distances[i - 1]) {
				currentK--;
				neighbours = new Instances(neighbours, 0, currentK);
				break;
			}
		}

		return neighbours;
	}

	/**
	 * Returns the revision string.
	 * 
	 * @return the revision
	 */
	public String getRevision() {
		return RevisionUtils.extract("$Revision: 10141 $");
	}

	/**
	 * Main method for testing this class.
	 *
	 * @param argv
	 *            should contain command line options (see setOptions)
	 */
	public static void main(String[] argv) {
		runClassifier(new IBk_JHSDV(), argv);
	}
}
