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

package ca.ubc.cs.beta.paramils.configurator;

import java.io.Serializable;
import java.text.DateFormat;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Date;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Set;
import java.util.Arrays;
import java.util.SortedMap;
import java.util.concurrent.atomic.AtomicBoolean;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.beust.jcommander.ParameterException;

import ca.ubc.cs.beta.aeatk.algorithmexecutionconfiguration.AlgorithmExecutionConfiguration;
import ca.ubc.cs.beta.aeatk.algorithmrunconfiguration.AlgorithmRunConfiguration;
import ca.ubc.cs.beta.aeatk.algorithmrunresult.AlgorithmRunResult;
import ca.ubc.cs.beta.aeatk.algorithmrunresult.RunStatus;
import ca.ubc.cs.beta.aeatk.eventsystem.EventManager;
import ca.ubc.cs.beta.aeatk.eventsystem.events.AutomaticConfiguratorEvent;
import ca.ubc.cs.beta.aeatk.eventsystem.events.ac.AutomaticConfigurationEnd;
import ca.ubc.cs.beta.aeatk.eventsystem.events.ac.IncumbentPerformanceChangeEvent;
import ca.ubc.cs.beta.aeatk.eventsystem.events.ac.IterationStartEvent;
import ca.ubc.cs.beta.aeatk.eventsystem.events.state.StateRestoredEvent;
import ca.ubc.cs.beta.aeatk.exceptions.DuplicateRunException;
import ca.ubc.cs.beta.aeatk.exceptions.OutOfTimeException;
import ca.ubc.cs.beta.aeatk.initialization.InitializationProcedure;
import ca.ubc.cs.beta.aeatk.misc.cputime.CPUTime;
import ca.ubc.cs.beta.aeatk.misc.watch.AutoStartStopWatch;
import ca.ubc.cs.beta.aeatk.misc.watch.StopWatch;
import ca.ubc.cs.beta.aeatk.objectives.RunObjective;
import ca.ubc.cs.beta.aeatk.objectives.OverallObjective;
import ca.ubc.cs.beta.aeatk.parameterconfigurationspace.ParameterConfiguration;
import ca.ubc.cs.beta.aeatk.parameterconfigurationspace.ParameterConfigurationSpace;
import ca.ubc.cs.beta.aeatk.parameterconfigurationspace.ParameterConfiguration.ParameterStringFormat;
import ca.ubc.cs.beta.aeatk.probleminstance.ProblemInstance;
import ca.ubc.cs.beta.aeatk.probleminstance.ProblemInstanceSeedPair;
import ca.ubc.cs.beta.aeatk.probleminstance.seedgenerator.InstanceSeedGenerator;
import ca.ubc.cs.beta.aeatk.random.RandomUtil;
import ca.ubc.cs.beta.aeatk.random.SeedableRandomPool;
import ca.ubc.cs.beta.aeatk.runhistory.RunHistory;
import ca.ubc.cs.beta.aeatk.state.StateDeserializer;
import ca.ubc.cs.beta.aeatk.state.StateFactory;
import ca.ubc.cs.beta.aeatk.state.StateSerializer;
import ca.ubc.cs.beta.aeatk.targetalgorithmevaluator.TargetAlgorithmEvaluator;
import ca.ubc.cs.beta.aeatk.termination.CompositeTerminationCondition;
import ca.ubc.cs.beta.aeatk.termination.TerminationCondition;
import ca.ubc.cs.beta.aeatk.termination.standard.ConfigurationSpaceExhaustedCondition;

import ca.ubc.cs.beta.paramils.misc.ParamILSOptions;
import ca.ubc.cs.beta.paramils.misc.ApproachMode;
import ca.ubc.cs.beta.paramils.misc.CappingMode;
import ca.ubc.cs.beta.paramils.misc.ConfigWriter;
import ca.ubc.cs.beta.paramils.misc.ExecutionMode;

public abstract class AbstractAlgorithmFramework {

  public ParamILSOptions options;
  public AlgorithmExecutionConfiguration execConfig;
  public List<ProblemInstance> instances;
  public TargetAlgorithmEvaluator tae;
  public ParameterConfigurationSpace configSpace;
  public InstanceSeedGenerator instanceSeedGen;
  public ParameterConfiguration initialIncumbent;
  public EventManager evtManager;
  public RunHistory runHistory;
  public SeedableRandomPool pool;
  public CompositeTerminationCondition termCond;
  public CPUTime cpuTime;

  public final double cutoffTime;
  public final String objectiveToReport;
  public final Logger log = LoggerFactory.getLogger(this.getClass());
  public static final DecimalFormat costdf = new DecimalFormat("#######.####");
  public static final DecimalFormat timedf = new DecimalFormat("######0.00");

  public int iteration = 0;
  public List<ProblemInstanceSeedPair> benchmark;
  public int bonusRuns;
  public String terminationReason = null;

  public TargetAlgorithmEvaluator vtae;
  public List<ProblemInstance> testInstances;
  public List<ProblemInstanceSeedPair> testBenchmark;

  public Map<ParameterConfiguration, Map<ProblemInstanceSeedPair, AlgorithmRunResult>> trainingHash = new HashMap<ParameterConfiguration, Map<ProblemInstanceSeedPair, AlgorithmRunResult>>();
  public Map<ParameterConfiguration, Map<ProblemInstanceSeedPair, AlgorithmRunResult>> validationHash = new HashMap<ParameterConfiguration, Map<ProblemInstanceSeedPair, AlgorithmRunResult>>();
  public int statNbRuns = 0;
  public int statNbUniqRuns = 0;
  public int statIncIterID = -1;

  public double infinity = Double.POSITIVE_INFINITY;

  public ConfigWriter configWriter;

  public AbstractAlgorithmFramework(ParamILSOptions paramilsOptions,
                                    AlgorithmExecutionConfiguration execConfig,
                                    List<ProblemInstance> instances,
                                    TargetAlgorithmEvaluator algoEval,
                                    ParameterConfigurationSpace configSpace,
                                    InstanceSeedGenerator instanceSeedGen,
                                    ParameterConfiguration initialIncumbent,
                                    EventManager manager,
                                    RunHistory rh,
                                    SeedableRandomPool pool,
                                    CompositeTerminationCondition termCond,
                                    CPUTime cpuTime) {
    this.options = paramilsOptions;
    this.execConfig = execConfig;
    this.instances = instances;
    this.tae = algoEval;
    this.configSpace = configSpace;
    this.instanceSeedGen = instanceSeedGen;
    this.initialIncumbent = initialIncumbent;
    this.evtManager = manager;
    this.runHistory = rh;
    this.pool = pool;
    this.termCond = termCond;
    this.cpuTime = cpuTime;

    if (initialIncumbent.isForbiddenParameterConfiguration()) {
      throw new ParameterException("Initial Incumbent specified is forbidden: " + this.initialIncumbent.getFormattedParameterString(ParameterStringFormat.NODB_SYNTAX));
    }

    for (Boolean continuous : configSpace.getContinuousMap().values())
      if (continuous)
        throw new ParameterException("Categorical parameters only");

    this.cutoffTime = paramilsOptions.scenarioConfig.algoExecOptions.cutoffTime;

    // TODO
    OverallObjective intraInstanceObj = paramilsOptions.scenarioConfig.getIntraInstanceObjective();
    switch(paramilsOptions.scenarioConfig.getRunObjective()) {
    case RUNTIME:
      switch(intraInstanceObj) {
      case MEAN:
        objectiveToReport = "mean runtime";
        break;
      case MEAN10:
        objectiveToReport = "penalized average runtime (PAR10)";
        break;
      case MEAN1000:
        objectiveToReport = "penalized average runtime (PAR1000)";
        break;
      default:
        objectiveToReport = intraInstanceObj + " " + paramilsOptions.scenarioConfig.getRunObjective();
      }
      break;
    case QUALITY:
      switch(intraInstanceObj) {
      case MEAN:
        objectiveToReport = "mean quality";
        break;
      default:
        objectiveToReport = intraInstanceObj + " " + paramilsOptions.scenarioConfig.getRunObjective();
        break;
      }
      break;
    default:
      objectiveToReport = intraInstanceObj + " " + paramilsOptions.scenarioConfig.getRunObjective();
      break;
    }

    // Clamping minRuns and maxRuns if necessary
    if (instanceSeedGen.getInitialInstanceSeedCount() < options.maxRuns) {
      options.maxRuns = instanceSeedGen.getInitialInstanceSeedCount();
      log.warn("MaxRuns clamped to {} due to lack of instance/seeds pairs", options.maxRuns);
    }
    if (options.minRuns > options.maxRuns)
      options.minRuns = options.maxRuns;

    benchmark = new ArrayList<ProblemInstanceSeedPair>();
    List<ProblemInstance> buff = new ArrayList<ProblemInstance>();
    while (benchmark.size() != options.maxRuns) {
      if (buff.size() == 0) {
        buff.addAll(instances);
        if (!options.deterministicInstanceOrdering) {
          Collections.shuffle(buff, pool.getRandom("PARAMILS_BENCHMARK_TRAINING"));
        }
      }
      ProblemInstance pi = buff.remove(0);
      synchronized (instanceSeedGen) {
        if (instanceSeedGen.hasNextSeed(pi)) {
          long seed = instanceSeedGen.getNextSeed(pi);
          benchmark.add(new ProblemInstanceSeedPair(pi, seed));
        }
      }
    }
  }

  public void fireEvent(AutomaticConfiguratorEvent evt) {
    this.evtManager.fireEvent(evt);
    this.evtManager.flush();
  }

  public void clean() {
    if (options.shutdownTAEWhenDone)
      tae.notifyShutdown();
  }

  /* ======================================================================
   * AEATK helper functions
   * ====================================================================== */

  public Map<ParameterConfiguration, Map<ProblemInstanceSeedPair, AlgorithmRunResult>> getHistoryHash() {
    if (vtae != null)
      return validationHash;
    else
      return trainingHash;
  }

  public ProblemInstanceSeedPair getBenchmark(int i) {
    if (vtae != null)
      return testBenchmark.get(i);
    else
      return benchmark.get(i);
  }

  public Map<ProblemInstanceSeedPair, AlgorithmRunResult> getRuns(ParameterConfiguration config) {
    Map<ProblemInstanceSeedPair, AlgorithmRunResult> h = getHistoryHash().get(config);
    if (h == null) {
      h = new HashMap<ProblemInstanceSeedPair, AlgorithmRunResult>();
      getHistoryHash().put(config, h);
    }
    return h;
  }

  public int getNbRunsFor(ParameterConfiguration config) {
    return getRuns(config).size();
  }

  public int getNbInstancesFor(ParameterConfiguration config) {
    Set<ProblemInstance> instances = new HashSet<>();
    for (ProblemInstanceSeedPair pisp : getRuns(config).keySet())
      instances.add(pisp.getProblemInstance());
    return instances.size();
  }

  public void runConfigOn(ParameterConfiguration config, ProblemInstanceSeedPair pisp, double cutoff) {
    // ensure SID
    configSIDC(config);
    // check termination
    if (vtae == null && termCond.haveToStop())
      throw new OutOfTimeException();
    // ensure a minimun runtime
    cutoff = Math.max(cutoff, 0.1);
    Map<ProblemInstanceSeedPair, AlgorithmRunResult> h = getHistoryHash().get(config);
    if (h == null) {
      h = new HashMap<ProblemInstanceSeedPair, AlgorithmRunResult>();
      getHistoryHash().put(config, h);
    } else {
      AlgorithmRunResult run = h.get(pisp);
      if (run != null)
        if (run.getRuntime() >= cutoff)
          throw new IllegalStateException("we are trying to run a configuration on an instance on which we already have better results");
    }
    AlgorithmRunConfiguration runConfig = new AlgorithmRunConfiguration(pisp, cutoff, config, execConfig);
    if (options.logDetails)
      log.info("{} run for {} on instance {} with seed {}", (vtae == null ? "Train" : "Test"), configSIDC(config), pisp.getProblemInstance().getInstanceName(), pisp.getSeed());
    List<AlgorithmRunResult> completedRuns = (vtae != null ? vtae : tae).evaluateRun(runConfig);
    for (AlgorithmRunResult run : completedRuns) {
      if (options.logDetails)
        log.info(" = \"{}\"", run.getResultLine());
      statNbRuns++;
      if (!h.containsKey(pisp))
        statNbUniqRuns++;
      h.put(pisp, run);
      try {
        runHistory.append(run);
      } catch (DuplicateRunException e) {
        // TODO: AEATK RunHistory is *FAR* too much SMAC-oriented
      }
    }
  }

  public void ensureConfigDetail(ParameterConfiguration config, int n) {
    int k = getNbRunsFor(config);
    if (k < n || (k == n && configWasCapped(config)))
      updateCache(config, n);
  }

  public void updateCache(ParameterConfiguration config, int n) {
    Map<ProblemInstanceSeedPair, AlgorithmRunResult> h = getRuns(config);
    ProblemInstanceSeedPair pisp;
    AlgorithmRunResult run;
    if (vtae == null)
      n = Math.min(Math.max(n, options.minRuns), options.maxRuns);
    for (int i=0; i<n; i++) {
      pisp = getBenchmark(i);
      run = h.get(pisp);
      if (run == null || (run.getRunStatus() == RunStatus.TIMEOUT && run.getRuntime() < cutoffTime))
        runConfigOn(config, pisp, cutoffTime);
    }
  }

  public int getTotalConfs() {
    return getHistoryHash().size();
  }

  public String getTunerTimeS() {
    double time = termCond.getTunerTime();
    int h = ((int) time)/3600;
    int m = ((int) time)/60%60;
    double s = time%60;
    return (h > 0 ? h + "h" + m + "m" : (m > 0 ? m + "m" : "")) + timedf.format(s) + "s";
  }

  public Boolean configWasCapped(ParameterConfiguration config) {
    int n = getNbRunsFor(config);
    if (n == 0)
      return false;
    AlgorithmRunResult lastRun = getRuns(config).get(benchmark.get(n-1));
    if (lastRun.getRunStatus() == RunStatus.TIMEOUT && lastRun.getAlgorithmRunConfiguration().getCutoffTime() < cutoffTime)
      return true; // capped on last run
    return false;
  }

  abstract public Boolean isIncumbent(ParameterConfiguration config);

  public String configSID(ParameterConfiguration config) {
    return configSIDC(config) + (isIncumbent(config) ? " (incumbent)" : "");
  }

  public String configSIDC(ParameterConfiguration config) {
    return "#" + runHistory.getOrCreateThetaIdx(config);
  }

  public String configSP(ParameterConfiguration config) {
    return "<"+config.getFormattedParameterString("-", " ", "", " ")+">";
    //return config.values().toString();
  }

  public void logConfig(String str, ParameterConfiguration config) {
    if (options.logDetails)
      log.info("({}) {}{} {}", getTunerTimeS(), str, configSID(config), configSP(config));
  }

  public String reportDiff(ParameterConfiguration init,
                           ParameterConfiguration dest,
                           String glue) {
    if (init == dest)
      return "no difference";
    StringBuilder sb = new StringBuilder();
    Boolean first = true;
    for (Entry<String, String> entry : init.entrySet()) {
      String key = entry.getKey();
      if (!init.get(key).equals(dest.get(key))) {
        if (first)
          first = false;
        else
          sb.append(glue);
        sb.append(key);
        sb.append(": ");
        sb.append(init.get(key));
        sb.append(" -> ");
        sb.append(dest.get(key));
      }
    }
    return sb.toString();
  }

  public String formatCost(double cost) {
    return costdf.format(cost);
  }

  /* ======================================================================
   * Main algorithm
   * ====================================================================== */

  public abstract void run();

  /* ======================================================================
   * Subfunctions
   * ====================================================================== */

  protected ParameterConfiguration randomConfig() {
    Random configSpaceRandom = pool.getRandom("PARAMILS_RANDOM_CONFIG");
    return configSpace.getRandomParameterConfiguration(configSpaceRandom);
  }

  public ArrayList<ParameterConfiguration> getNeighborhood(ParameterConfiguration config) {
    Random configRandN = pool.getRandom("PARAMILS_RANDOM_NEIGHBOR_CONT");
    ArrayList<ParameterConfiguration> pop = new ArrayList<>(config.getNeighbourhood(configRandN, options.scenarioConfig.algoExecOptions.paramFileDelegate.continuousNeighbours));
    return pop;
  }

  protected ParameterConfiguration randomNeighbor(ParameterConfiguration config) {
    Random configRandN = pool.getRandom("PARAMILS_RANDOM_NEIGHBOR");
    ArrayList<ParameterConfiguration> pop = getNeighborhood(config);
    Collections.shuffle(pop, configRandN);
    return pop.get(0);
  }

  // gets random neighbor from priority space
  protected ParameterConfiguration randomNeighborPriority(ParameterConfiguration config) {
    Random configRandN = pool.getRandom("PARAMILS_RANDOM_NEIGHBOR");
    ArrayList<ParameterConfiguration> pop = getNeighborhood(config);
    Collections.shuffle(pop, configRandN);
    ParameterConfiguration nextConfig = pop.get(0);
    log.info("Value array: {}", Arrays.toString(nextConfig.toValueArray()));
    Set<String> set = nextConfig.keySet();
    String[] myArray = new String[set.size()];
    set.toArray(myArray);
    log.info("Key Set: {}", Arrays.toString(myArray));
    return nextConfig;
  }

  int stagCounter = 0;
  double lastTunerTime = -1;
  protected void checkTunerTime() {
    double tunerTime = termCond.getTunerTime();
    if (lastTunerTime == tunerTime)
      stagCounter++;
    else
      stagCounter = 0;
    lastTunerTime = tunerTime;
    //log.info("{} cache-only tests", stagCounter);
    if (stagCounter > configSpace.getUpperBoundOnSize()) {
      terminationReason = "Premature termination (after " + getTunerTimeS() + ")"; // TODO
      throw new OutOfTimeException(); // TODO: ugly: why are exceptions handled that way?
    }
  }

  /* ======================================================================
   * Final Report
   * ====================================================================== */

  public String getTerminationReason() {
    return terminationReason;
  }

  public String getCallString(ParameterConfiguration config) {
    ProblemInstanceSeedPair pisp = benchmark.get(0);
    AlgorithmRunConfiguration runConfig = new AlgorithmRunConfiguration(pisp, cutoffTime, config, execConfig);
    return tae.getManualCallString(runConfig);
  }

  public abstract void logFinalReport();

  /* ======================================================================
   * Config Writer
   * ====================================================================== */

  public void setupConfigWriter(ConfigWriter cw) {
    this.configWriter = cw;
  }

  public void logConfigOnFile(ParameterConfiguration config) {
    if (configWriter != null)
      configWriter.writeConfig(config);
    else
      throw new IllegalStateException("ConfigWriter not set (yet)");
  }

  /* ======================================================================
   * Validation
   * ====================================================================== */

  public void validate(TargetAlgorithmEvaluator validatingTae,
                       List<ProblemInstance> testInstances,
                       InstanceSeedGenerator testInstanceSeedGen) {
    this.vtae = validatingTae;
    this.testInstances = testInstances;

    testBenchmark = new ArrayList<ProblemInstanceSeedPair>();
    List<ProblemInstance> buff = new ArrayList<ProblemInstance>();
    int max_runs = testInstanceSeedGen.getInitialInstanceSeedCount();
    if (max_runs < options.validationRuns) {
      log.warn("Clamping number of test runs to {} due to lack of instance/seeds pairs", max_runs);
      options.validationRuns = max_runs;
    } else {
      max_runs = options.validationRuns;
    }

    while (testBenchmark.size() != max_runs) {
      if (buff.size() == 0) {
        buff.addAll(testInstances);
        //Collections.shuffle(buff, pool.getRandom("PARAMILS_BENCHMARK_TEST")); // dont shuffle: wrong shuffle pool
      }
      ProblemInstance pi = buff.remove(0);
      synchronized (testInstanceSeedGen) {
        if (testInstanceSeedGen.hasNextSeed(pi)) {
          long seed = testInstanceSeedGen.getNextSeed(pi);
          testBenchmark.add(new ProblemInstanceSeedPair(pi, seed));
        }
      }
    }

    doValidation();
  }

  public abstract void doValidation();

  /* ======================================================================
   * Final Validation Report
   * ====================================================================== */

  public abstract void logValidationMessage();
}
