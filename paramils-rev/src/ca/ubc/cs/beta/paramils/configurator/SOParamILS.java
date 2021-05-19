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

import ca.ubc.cs.beta.paramils.misc.ParamILSOptions;
import ca.ubc.cs.beta.paramils.misc.ApproachMode;
import ca.ubc.cs.beta.paramils.misc.CappingMode;
import ca.ubc.cs.beta.paramils.misc.ExecutionMode;

public abstract class SOParamILS extends AbstractAlgorithmFramework {

  public ParameterConfiguration incumbent = null;
  public ParameterConfiguration current, random, tmp;

  public double cost_incumbent = infinity;
  public int cost_incumbent_n = 0;

  public double testInitialCost;
  public double testFinalCost;

  /* ======================================================================
   * Constructor
   * ====================================================================== */

  public SOParamILS(ParamILSOptions paramilsOptions,
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
    super(paramilsOptions, execConfig, instances, algoEval, configSpace, instanceSeedGen, initialIncumbent, manager, rh, pool, termCond, cpuTime);

    // Checking capping options
    switch (options.cappingMode) {
    case AUTO:
      switch (options.scenarioConfig.getRunObjective()) {
      case RUNTIME:
        options.cappingMode = CappingMode.FINE;
        break;
      case QUALITY:
        options.cappingMode = CappingMode.OFF;
        break;
      default:
        throw new IllegalStateException("Not sure what to default to");
      }
      break;
    case FINE:
      switch (options.scenarioConfig.getRunObjective()) {
      case QUALITY:
        throw new ParameterException("Fine capping impossible for quality optimization");
      }
      break;
    }

    switch (options.scenarioConfig.getRunObjective()) {
    case RUNTIME:
      if (options.cappingMinValue < 0)
        throw new ParameterException("Theoretical minimum runtime can't be negative");
      break;
    }
  }

  /* ======================================================================
   * AEATK helper functions
   * ====================================================================== */

  public void updateCacheWithBudget(ParameterConfiguration config, int k, double max_budget) {
    OverallObjective obj = options.scenarioConfig.getIntraInstanceObjective(); // mean
    RunObjective runobj = options.scenarioConfig.getRunObjective(); // runtime
    Map<ProblemInstanceSeedPair, AlgorithmRunResult> h;
    AlgorithmRunResult run;
    double budget = max_budget;
    double cutoff = cutoffTime;
    if (vtae == null)
      k = Math.min(Math.max(k, options.minRuns), options.maxRuns);
    for (int i=0; i<k; i++) {
      ProblemInstanceSeedPair pisp = getBenchmark(i);
      switch (runobj) {
      case RUNTIME:
        switch (options.cappingMode) {
        case BASIC:
          cutoff = cutoffTime;
          break;
        case FINE:
          cutoff = Math.min(budget, cutoffTime);
          break;
        }
        break;
      case QUALITY:
        cutoff = cutoffTime;
      }
      h = getRuns(config);
      run = h.get(pisp);
      if (run == null || (run.getRunStatus() == RunStatus.TIMEOUT && cutoff-run.getRuntime() > options.cutoffThreshold*cutoffTime)) {
        runConfigOn(config, pisp, cutoff);
        run = h.get(pisp);
      }
      switch (runobj) {
      case RUNTIME:
        if (run.getRunStatus() == RunStatus.TIMEOUT)
          budget -= obj.getPenaltyFactor()*run.getRuntime();
        else
          budget -= run.getRuntime();
        break;
      case QUALITY:
        budget -= run.getQuality();
        break;
      }
      if (budget < (k-i)*options.cappingMinValue)
        break;
    }
  }

  public double minTimeout(ParameterConfiguration config, int k, double cutoff) {
    Map<ProblemInstanceSeedPair, AlgorithmRunResult> h = getRuns(config);
    ProblemInstanceSeedPair pisp;
    AlgorithmRunResult run;
    double min_timeout = cutoff;
    double timeout;
    for (int i=0; i<k; i++) {
      pisp = getBenchmark(i);
      run = h.get(pisp);
      if (run != null) {
        if (run.getRunStatus() == RunStatus.TIMEOUT) {
          timeout = run.getAlgorithmRunConfiguration().getCutoffTime();
          min_timeout = Math.min(min_timeout, timeout);
        }
      }
    }
    return min_timeout;
  }

  public double computeConfigCost(ParameterConfiguration config, int k) {
    return computeConfigCost(config, k, cutoffTime);
  }

  public double computeConfigCost(ParameterConfiguration config, int k, double cutoff) {
    OverallObjective obj = options.scenarioConfig.getIntraInstanceObjective(); // mean
    RunObjective runobj = options.scenarioConfig.getRunObjective(); // runtime
    Map<ProblemInstanceSeedPair, AlgorithmRunResult> h = getRuns(config);
    ProblemInstanceSeedPair pisp;
    AlgorithmRunResult run;
    Collection<Double> l = new ArrayList<Double>();
    double min_timeout = minTimeout(config, k, cutoff);
    double timeout;
    for (int i=0; i<k; i++) {
      pisp = getBenchmark(i);
      run = h.get(pisp);
      if (run == null)
        return infinity; // insufficient runs
      else
        l.add(runobj.getObjective(run));
    }
    double cost = obj.aggregate(l, min_timeout);
    return cost;
  }

  public double computeConfigCostWithBudget(ParameterConfiguration config, int k, double budget) {
    OverallObjective obj = options.scenarioConfig.getIntraInstanceObjective(); // mean
    RunObjective runobj = options.scenarioConfig.getRunObjective(); // runtime
    Map<ProblemInstanceSeedPair, AlgorithmRunResult> h = getRuns(config);
    ProblemInstanceSeedPair pisp;
    AlgorithmRunResult run;
    Collection<Double> l = new ArrayList<Double>();
    double min_timeout = minTimeout(config, k-1, cutoffTime); // huh, k-1
    double cost = infinity;
    for (int i=0; i<k; i++) {
      pisp = getBenchmark(i);
      run = h.get(pisp);
      if (run == null)
        return infinity; // config has been capped
      else
        l.add(runobj.getObjective(run));
      cost = obj.aggregate(l, min_timeout);
      if (cost > budget)
        return infinity; // over budget
    }
    return cost;
  }

  public double configCost(ParameterConfiguration config, int k) {
    if (k == 0)
      return infinity;
    // update cache
    updateCache(config, k);
    // compute cost
    int n = getNbRunsFor(config);
    if (n < k)
      return infinity;
    return computeConfigCost(config, k);
  }

  public double configCostWithBudget(ParameterConfiguration config, int k, double budget) {
    if (k == 0)
      return infinity;
    // update cache
    updateCacheWithBudget(config, k, budget);
    // compute cost
    int n = getNbRunsFor(config);
    if (n < k || configWasCapped(config))
      return infinity;
    return computeConfigCostWithBudget(config, k, budget);
  }

  public void updateIncumbent() {
    int n;
    double cost;
    n = getNbRunsFor(incumbent);
    if (n <= cost_incumbent_n)
      return;
    cost = computeConfigCost(incumbent, n);
    fireEvent(new IncumbentPerformanceChangeEvent(termCond, cost, incumbent, n, incumbent, cpuTime));
    log.info("Config {} updated!", configSID(incumbent));
    log.info(" {} -> {} (based on {} and {} runs)", costdf.format(cost_incumbent), costdf.format(cost), cost_incumbent_n, n);
    cost_incumbent = cost;
    cost_incumbent_n = n;
  }

  public void checkIncumbent(ParameterConfiguration config) {
    updateIncumbent();
    if (!isIncumbent(config)) {
      int n = getNbRunsFor(config);
      ensureConfigDetail(incumbent, n);
      updateIncumbent();
      double cost = computeConfigCost(config, n);
      if (!dominates(config, incumbent))
        return;
      fireEvent(new IncumbentPerformanceChangeEvent(termCond, cost, config, n, incumbent, cpuTime));
      log.info("New incumbent! ({})", configSID(config));
      log.info(" {}", reportDiff(incumbent, config, " ; "));
      log.info(" {} -> {} (based on {} and {} runs)", costdf.format(cost_incumbent), costdf.format(cost), cost_incumbent_n, n);
      incumbent = config;
      statIncIterID = iteration;
      cost_incumbent = cost;
      cost_incumbent_n = n;
    }
  }

  public Boolean isIncumbent(ParameterConfiguration config) {
    return config.equals(incumbent);
  }

  /* ======================================================================
   * Main algorithm
   * ====================================================================== */

  public void run() {
    long time = System.currentTimeMillis();
    Date d = new Date(time);
    DateFormat df = DateFormat.getDateTimeInstance();
    log.info("ParamILS started at: {}. Minimizing {}.", df.format(d), objectiveToReport);
    try {
      iteration = 0;
      incumbent = initialIncumbent;
      beforeMainLoop(initialIncumbent);
      while (!termCond.haveToStop()) {
        iteration++;
        log.info("== Starting Iteration {} ==", iteration);
        log.info("Tuner time: {} elapsed", getTunerTimeS());
        insideMainLoop(iteration);
      }
    } catch (OutOfTimeException e) {
      checkIncumbent(incumbent);
    }
    if (terminationReason == null)
      terminationReason = termCond.getTerminationReason();
    log.info("ParamILS completed");
  }

  abstract void beforeMainLoop(ParameterConfiguration init);

  abstract void insideMainLoop(int iteration);

  /* ======================================================================
   * Subfunctions
   * ====================================================================== */

  protected ParameterConfiguration localSearch(ParameterConfiguration init) {
    ParameterConfiguration best = init;
    ParameterConfiguration local = init;
    Random configRandLS = pool.getRandom("PARAMILS_LOCAL_SEARCH_NEIGHBOURS");
    List<ParameterConfiguration> neighbourhood;
    List<ParameterConfiguration> tabuList = new ArrayList<ParameterConfiguration>();
    tabuList.add(init);
    while (true) {
      local = best;
      neighbourhood = local.getNeighbourhood(configRandLS, 0);
      Collections.shuffle(neighbourhood, configRandLS);
      for (ParameterConfiguration neighbour : neighbourhood) {
        if (options.logDetails)
          logConfig("trying ", neighbour);
        if (tabuList.contains(neighbour)) {
          if (options.logDetails)
            log.info(" ... tabu");
        } else {
          checkTunerTime();
          tabuList.add(neighbour);
          if (better(neighbour, local)) {
            if (options.logDetails) {
              int k1 = getNbRunsFor(neighbour);
              int k2 = getNbRunsFor(local);
              log.info(" ... new local best!");
              log.info(" {} -> {} (based on {} and {} runs)", formatCost(computeConfigCost(local, k2)), formatCost(computeConfigCost(neighbour, k1)), k2, k1);
            }
            best = neighbour;
            checkIncumbent(best);
            break;
          } else {
            if (options.logDetails) {
              int k1 = getNbRunsFor(neighbour);
              int k2 = getNbRunsFor(local);
              if (k1 < k2) {
                log.info(" ... capped after {}/{} runs", k1, k2);
              } else {
                log.info(" ... worse");
              }
            }
          }
        }
      }
      if (local.equals(best)) {
        break;
      }
    }
    if (options.logDetails)
      log.info("-- local search completed --");
    return best;
  }

  protected Boolean dominates(ParameterConfiguration a, ParameterConfiguration b) {
    int nA = getNbRunsFor(a);
    int nB = getNbRunsFor(b);
    // TODO: can probably be better
    if (configWasCapped(a))
      nA -= 1;
    if (configWasCapped(b))
      nB -= 1;
    if (nA < nB)
      return false;
    return computeConfigCost(a, nB) <= computeConfigCost(b, nB);
  }

  abstract Boolean better(ParameterConfiguration a, ParameterConfiguration b);

  /* ======================================================================
   * Final Report
   * ====================================================================== */

  public void logFinalReport() {
    final DecimalFormat df0 = new DecimalFormat("0");
    log.info("Total number of runs performed: {} ({}), total configurations tried: {}", statNbRuns, statNbUniqRuns, getTotalConfs());
    log.info("Total CPU time used: {} s, total wallclock time used: {} s", df0.format(termCond.getTunerTime()), df0.format(termCond.getWallTime()));
    log.info("ParamILS's final incumbent found iteration {}, ID #{}", statIncIterID, runHistory.getThetaIdx(incumbent));
    log.info("Estimated {} of final incumbent on training set: {}", objectiveToReport, formatCost(cost_incumbent));
    log.info("Estimation based on {} run{} on {} training instance{}", cost_incumbent_n, (cost_incumbent_n > 1 ? "s" : ""), getNbInstancesFor(incumbent), (getNbInstancesFor(incumbent) > 1 ? "s" : ""));
    if (!options.doValidation)
      log.info("Sample call for final incumbent:\n{}", getCallString(initialIncumbent));
    logConfigOnFile(incumbent);
    if (initialIncumbent != incumbent) {
      log.info("------------------------------------------------------------------------");
      log.info("Differences with initial configuration: \n " + reportDiff(initialIncumbent, incumbent, "\n "));
    }
  }

  /* ======================================================================
   * Validation
   * ====================================================================== */

  public void doValidation() {
    int n = options.validationRuns;
    log.info("Testing final incumbent");
    updateCache(incumbent, n);
    testFinalCost = computeConfigCost(incumbent, n);
    if (options.compareInitialConfiguration) {
      log.info("Testing initial config");
      updateCache(initialIncumbent, n);
      testInitialCost = computeConfigCost(initialIncumbent, n);
    }
  }

  /* ======================================================================
   * Final Validation Report
   * ====================================================================== */

  public void logValidationMessage() {
    log.info("Estimated {} of final incumbent on test set: {}", objectiveToReport, formatCost(testFinalCost));
    log.info("Estimation based on {} run{} on {} test instance{}", testBenchmark.size(), (testBenchmark.size() > 1 ? "s" : ""), testInstances.size(), (testInstances.size() > 1 ? "s" : ""));
    log.info("Sample call for final incumbent:\n{}", getCallString(incumbent));
    if (options.compareInitialConfiguration) {
      log.info("------------------------------------------------------------------------");
      log.info("Estimated {} of initial configuration on test set: {}", objectiveToReport, formatCost(testInitialCost));
      String gain;
      if (testInitialCost > 0 && testFinalCost > 0)
        gain = formatCost(testInitialCost/testFinalCost);
      else if (testInitialCost < 0 && testFinalCost < 0)
        gain = formatCost(testFinalCost/testInitialCost);
      else
        gain = "unclear.";
      log.info("Cost gain for this run is {}", gain);
      log.info("Sample call for initial configuration:\n{}", getCallString(initialIncumbent));
    }
  }
}
