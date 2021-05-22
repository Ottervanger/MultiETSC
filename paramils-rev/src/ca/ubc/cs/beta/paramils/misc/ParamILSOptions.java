/*
  Copyright (C) UBC, Vancouver; CRIStAL, Lille, 2016-2017
  Aymeric Blot

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

package ca.ubc.cs.beta.paramils.misc;

import java.io.File;
import java.io.IOException;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.ArrayList;

import ca.ubc.cs.beta.aeatk.acquisitionfunctions.AcquisitionFunctions;
import ca.ubc.cs.beta.aeatk.algorithmexecutionconfiguration.AlgorithmExecutionConfiguration;
import ca.ubc.cs.beta.aeatk.help.HelpOptions;
import ca.ubc.cs.beta.aeatk.logging.ComplexLoggingOptions;
import ca.ubc.cs.beta.aeatk.misc.file.HomeFileUtils;
import ca.ubc.cs.beta.aeatk.misc.jcommander.validator.*;
import ca.ubc.cs.beta.aeatk.misc.options.CommandLineOnly;
import ca.ubc.cs.beta.aeatk.misc.options.OptionLevel;
import ca.ubc.cs.beta.aeatk.misc.options.UsageTextField;
import ca.ubc.cs.beta.aeatk.model.ModelBuildingOptions;
import ca.ubc.cs.beta.aeatk.options.AbstractOptions;
import ca.ubc.cs.beta.aeatk.options.RandomForestOptions;
import ca.ubc.cs.beta.aeatk.options.RunGroupOptions;
import ca.ubc.cs.beta.aeatk.options.scenario.ScenarioOptions;
import ca.ubc.cs.beta.aeatk.parameterconfigurationspace.ParameterConfigurationSpace;
import ca.ubc.cs.beta.aeatk.parameterconfigurationspace.tracking.ParamConfigurationOriginTrackingOptions;
import ca.ubc.cs.beta.aeatk.probleminstance.InstanceListWithSeeds;
import ca.ubc.cs.beta.aeatk.probleminstance.ProblemInstance;
import ca.ubc.cs.beta.aeatk.probleminstance.ProblemInstanceOptions.TrainTestInstances;
import ca.ubc.cs.beta.aeatk.random.SeedOptions;
import ca.ubc.cs.beta.aeatk.random.SeedableRandomPool;
import ca.ubc.cs.beta.aeatk.random.SeedableRandomPoolConstants;

import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterFile;
import com.beust.jcommander.ParametersDelegate;


/**
 * Represents the configuration for ParamILS,
 */

@UsageTextField(title="ParamILS Options", description="General Options for Running ParamILS", claimRequired={"--pcs-file","--instanceFile","--run-obj"}, noarg=ParamILSNoArgHandler.class)
public class ParamILSOptions extends AbstractOptions {

  /* ======================================================================
   * ParamILS options
   * ====================================================================== */

  @UsageTextField(level=OptionLevel.BASIC)
  @Parameter(names={"--MO","--multiObjective"}, description="multi-objective tuning")
  public Boolean multiObjective = false;

  @UsageTextField(level=OptionLevel.BASIC)
  @Parameter(names={"--initial-incumbent","--initialIncumbent"}, description="Initial Incumbent to use for configuration (you can use RANDOM, FILE, or DEFAULT as a special string to get a RANDOM or the DEFAULT configuration as needed). Other configurations are specified as: -name 'value' -name 'value' ... For instance: --quick-sort 'on'.")
  public String initialIncumbent = "DEFAULT";

  @UsageTextField(level=OptionLevel.BASIC)
  @Parameter(names={"--initial-incumbent-file","--initialIncumbentFile"}, description="If initialIncumbent is FILE, reads the configurations stored in the specified file. In single objective mode, only the last one is used")
  public String initialIncumbentFile = "";

  @UsageTextField(level=OptionLevel.INTERMEDIATE)
  @Parameter(names={"--min-runs","--minRuns"}, description="minimum number of runs required", validateWith=FixedPositiveInteger.class)
  public int minRuns = 1;

  @UsageTextField(level=OptionLevel.BASIC)
  @Parameter(names={"--max-runs","--maxRuns","--N"}, description="maximum number of runs allowed", validateWith=FixedPositiveInteger.class)
  public int maxRuns = 2000;

  @UsageTextField(level=OptionLevel.INTERMEDIATE)
  @Parameter(names={"--R"}, description="amount of random configurations considered to get the first incumbent")
  public int R = 10;

  @UsageTextField(level=OptionLevel.ADVANCED)
  @Parameter(names={"--n-local-search"}, description="The maximum number of neighbours searched during local seach")
  public int nLocalSearch = 0;

  @UsageTextField(level=OptionLevel.ADVANCED)
  @Parameter(names={"--perturb"}, description="perturbation length between local searches")
  public int perturbationLength = 3;

  @UsageTextField(level=OptionLevel.INTERMEDIATE)
  @Parameter(names={"--randomRestart","--random-restart"}, description="random restart probability")
  public double randomRestartProbability = 0.05;

  @UsageTextField(level=OptionLevel.BASIC)
  @Parameter(names={"--approach"}, description="Comparaison approach of ParamILS")
  public ApproachMode approach = ApproachMode.FOCUSED;

  @UsageTextField(level=OptionLevel.ADVANCED)
  @Parameter(names={"--bonus-runs-mode","--bonusRunsMode"}, description="Bonus run mechanism of FocusILS")
  public BonusRunsMode bonusRunsMode = BonusRunsMode.INTENSIFY;

  @UsageTextField(level=OptionLevel.ADVANCED)
  @Parameter(names={"--n-tradeoff-params"}, description="The number of parameters dedicated to controlling the MO trade-off. The first n parameters in the pcs-file are searched more exploratorive.")
  public int nTradeoffParams = 0;
  
  public ArrayList<String> tradeoffParams;

  /* ======================================================================
   * Capping options
   * ====================================================================== */

  @UsageTextField(level=OptionLevel.INTERMEDIATE)
  @Parameter(names={"--capping-mode","--cappingMode"}, description="Capping mode of ParamILS")
  public CappingMode cappingMode = CappingMode.AUTO;

  @UsageTextField(level=OptionLevel.INTERMEDIATE)
  @Parameter(names={"--aggressive-capping","--aggressiveCapping"}, description="Use Aggressive Capping")
  public Boolean aggressiveCapping = true;

  @UsageTextField(level=OptionLevel.ADVANCED)
  @Parameter(names={"--aggressive-capping-factor","--aggressiveCappingFactor"}, description="Aggressive Capping factor")
  public double aggressiveCappingFactor = 2;

  @UsageTextField(level=OptionLevel.ADVANCED)
  @Parameter(names={"--cutoff-threshold","--cutoffThreshold"}, description="Minimum percentage of the default cutoff time required before resheduling a run")
  public double cutoffThreshold = 0.1;

  @UsageTextField(level=OptionLevel.INTERMEDIATE)
  @Parameter(names={"--capping-min-value","--cappingMinValue"}, description="Minimum value for capping estimation")
  public double cappingMinValue = 0;

  /* ======================================================================
   * Validation options
   * ====================================================================== */

  @CommandLineOnly
  @UsageTextField(level=OptionLevel.BASIC)
  @Parameter(names={"--validation","--doValidation"}, description="perform validation when ParamILS completes")
  public boolean doValidation = true;

  @CommandLineOnly
  @UsageTextField(level=OptionLevel.BASIC)
  @Parameter(names={"--validationRuns","--validation-runs"}, description = "Number of validation runs (if enough entries in test instance file). To disable validation see the --doValidation option", validateWith=FixedPositiveInteger.class)
  public int validationRuns = 1000;

  @UsageTextField(level=OptionLevel.BASIC)
  @Parameter(names={"--compare-initial-config","--compareInitialConfiguration"}, description="if true we will also test the initial config during validation")
  public boolean compareInitialConfiguration = false;

  @UsageTextField(level=OptionLevel.INTERMEDIATE)
  @Parameter(names={"--validate-on-training-set","--validateOnTrainingSet"}, description="Validation will reuse the training set")
  public boolean validateOnTrainingSet = false;

  /* ======================================================================
   * Others options
   * ====================================================================== */

  @UsageTextField(level=OptionLevel.BASIC)
  @Parameter(names={"--log-details","--logDetails"}, description="Print full details of the search")
  public Boolean logDetails = false;

  @UsageTextField(level=OptionLevel.INTERMEDIATE)
  @Parameter(names={"--always-run-initial-config","--alwaysRunInitialConfiguration"}, description="if true we will always run the default and switch back to it if it is better than the incumbent")
  public boolean alwaysRunInitialConfiguration = false;

  @UsageTextField(level=OptionLevel.INTERMEDIATE)
  @Parameter(names={"--deterministic-instance-ordering","--deterministicInstanceOrdering"}, description="If true, instances will be selected from the instance list file in the specified order")
  public boolean deterministicInstanceOrdering = false;

  @CommandLineOnly
  @UsageTextField(defaultValues="<current working directory>", level=OptionLevel.INTERMEDIATE)
  @Parameter(names={"--experiment-dir","--experimentDir","-e"}, description="root directory for experiments Folder")
  public String experimentDir = System.getProperty("user.dir") + File.separator + "";

  @ParametersDelegate
  public HelpOptions help = new HelpOptions();

  @ParametersDelegate
  public ComplexLoggingOptions logOptions = new ComplexLoggingOptions();

  @UsageTextField(defaultValues="", level=OptionLevel.ADVANCED)
  @ParameterFile
  @Parameter(names={"--option-file","--optionFile"}, description="read options from file")
  public File optionFile;

  @UsageTextField(defaultValues="", level=OptionLevel.ADVANCED)
  @ParameterFile
  @Parameter(names={"--option-file2","--optionFile2","--secondaryOptionsFile"}, description="read options from file")
  public File optionFile2;

  @ParametersDelegate
  public RunGroupOptions runGroupOptions = new RunGroupOptions("%SCENARIO_NAME");

  @ParametersDelegate
  public ScenarioOptions scenarioConfig = new ScenarioOptions();

  @ParametersDelegate
  public SeedOptions seedOptions = new SeedOptions();

  @UsageTextField(defaultValues="~/.aeatk/paramils.opt", level=OptionLevel.ADVANCED)
  @Parameter(names={"--paramils-default-file","--paramilsDefaultsFile"}, description="file that contains default settings for ParamILS")
  @ParameterFile(ignoreFileNotExists = true)
  public File paramilsDefaults = HomeFileUtils.getHomeFile(".aeatk" + File.separator  + "paramils.opt");

  @ParametersDelegate
  public ParamConfigurationOriginTrackingOptions trackingOptions= new ParamConfigurationOriginTrackingOptions();

  @UsageTextField(defaultValues="0 which should cause it to run exactly the same as the stand-alone utility.", level=OptionLevel.ADVANCED)
  @Parameter(names="--validation-seed", description="Seed to use for validating ParamILS")
  public int validationSeed = 0;

  public boolean shutdownTAEWhenDone = true;

  public ParamILSOptions() {
    // because the default in AEATK is shamefully "smac-output"
    scenarioConfig.outputDirectory = "paramils-output";
    // overhead not included in tunertime
    scenarioConfig.limitOptions.countSMACTimeAsTunerTime = false;
  }

  public AlgorithmExecutionConfiguration getAlgorithmExecutionConfig() {
    return this.scenarioConfig.getAlgorithmExecutionConfig(experimentDir);
  }

  public String getOutputDirectory(String runGroupName) {
    File outputDir = new File(this.scenarioConfig.outputDirectory + File.separator + runGroupName);
    if(!outputDir.isAbsolute()) {
      outputDir = new File(experimentDir + File.separator + this.scenarioConfig.outputDirectory + File.separator + runGroupName);
    }
    return outputDir.getAbsolutePath();
  }

  public String getRunGroupName(Collection<AbstractOptions> opts) {
    opts = new HashSet<AbstractOptions>(opts);
    opts.add(this);
    return runGroupOptions.getRunGroupName(opts);
  }

  /**
   * Gets both the training and the test problem instances
   *
   * @param experimentDirectory Directory to search for instance files
   * @param trainingSeed      Seed to use for the training instances
   * @param testingSeed     Seed to use for the testing instances
   * @param trainingRequired    Whether the training instance file is required
   * @param testRequired      Whether the test instance file is required
   * @return
   * @throws IOException
   */
  public TrainTestInstances getTrainingAndTestProblemInstances(SeedableRandomPool instancePool, SeedableRandomPool testInstancePool) throws IOException {
    return this.scenarioConfig.getTrainingAndTestProblemInstances(this.experimentDir, instancePool.getRandom(SeedableRandomPoolConstants.INSTANCE_SEEDS).nextInt(), testInstancePool.getRandom(SeedableRandomPoolConstants.TEST_SEED_INSTANCES).nextInt(), true, this.doValidation, false, false);
  }
}
