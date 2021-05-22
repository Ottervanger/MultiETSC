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
import java.util.Arrays;
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

import java.lang.reflect.Method;

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

public abstract class MOParamILS extends AbstractAlgorithmFramework {

  public ArrayList<ParameterConfiguration> incumbentArchive;
  public ArrayList<ParameterConfiguration> currentArchive;
  public ArrayList<ParameterConfiguration> validationArchive;
  public ParameterConfiguration current, random, tmp;

  public ArrayList<ParameterConfiguration> initialIncumbentList;

  public Map<ParameterConfiguration, ArrayList<Double> > validationHash = new HashMap<>();

  int number_of_objectives = 0;
  ArrayList<Double> infinity_cost;

  /* ======================================================================
   * Constructor
   * ====================================================================== */

  public MOParamILS(ParamILSOptions paramilsOptions,
                    AlgorithmExecutionConfiguration execConfig,
                    List<ProblemInstance> instances,
                    TargetAlgorithmEvaluator algoEval,
                    ParameterConfigurationSpace configSpace,
                    InstanceSeedGenerator instanceSeedGen,
                    ArrayList<ParameterConfiguration> initialIncumbentList,
                    EventManager manager,
                    RunHistory rh,
                    SeedableRandomPool pool,
                    CompositeTerminationCondition termCond,
                    CPUTime cpuTime) {
    super(paramilsOptions, execConfig, instances, algoEval, configSpace, instanceSeedGen, initialIncumbentList.get(0), manager, rh, pool, termCond, cpuTime);

    this.initialIncumbentList = initialIncumbentList;

    switch (options.scenarioConfig.getRunObjective()) {
    case RUNTIME:
      if (configSpace.getValuesMap().get("runtime") == null)
        throw new ParameterException("Cannot optimize RUNTIME runObj without a --runtime parameter");
      break;
    }

    // Checking capping options
    switch (options.cappingMode) {
    case AUTO:
      options.cappingMode = CappingMode.BASIC;
      break;
    case FINE:
      throw new ParameterException("Fine capping not applicable for multi objective optimization");
    }

    // TODO
    switch (options.scenarioConfig.getRunObjective()) {
    case RUNTIME:
      //if (options.cappingMinValue < 0)
      //  throw new ParameterException("Theoretical minimum runtime can't be negative");
      break;
    }

  }

  /* ======================================================================
   * AEATK helper functions
   * ====================================================================== */

  public ArrayList<Double> computeConfigCostFromList(ArrayList<ArrayList<Double> > l) {
    if (l.size() == 0) // should hopefully never happen
      return infinity_cost;
    ArrayList<Double> cost = new ArrayList<>(l.get(0));
    int m = cost.size();
    int k = l.size();
    for (int i=1; i<k; i++) {
      ArrayList<Double> c = l.get(i);
      if (c.size() != m)
        throw new IllegalStateException("Inconsistent number of objectives");
      for (int j=0; j<m; j++)
        cost.set(j, cost.get(j) + c.get(j));
    }
    for (int j=0; j<m; j++)
      cost.set(j, cost.get(j)/k);
    return cost;
  }

  public ArrayList<Double> computeConfigCost(ParameterConfiguration config) {
    return computeConfigCost(config, getNbRunsFor(config));
  }

  public ArrayList<Double> computeConfigCost(ParameterConfiguration config, int k) {
    OverallObjective obj = options.scenarioConfig.getIntraInstanceObjective(); // mean
    RunObjective runobj = options.scenarioConfig.getRunObjective(); // runtime
    Map<ProblemInstanceSeedPair, AlgorithmRunResult> h = getRuns(config);
    ProblemInstanceSeedPair pisp;
    AlgorithmRunResult run;
    ArrayList<ArrayList<Double >> l = new ArrayList<ArrayList<Double> >();
    for (int i=0; i<k; i++) {
      pisp = getBenchmark(i);
      run = h.get(pisp);
      if (run == null || run.getRunStatus().equals(RunStatus.CRASHED)
          || run.getRunStatus().equals(RunStatus.KILLED))
        if (infinity_cost == null)
          throw new IllegalStateException("The very first run was faulty. Unacceptable.");
        else
          return infinity_cost; // insufficient runs, or problematic run found
      else
        l.add(run.getQualityArray());
    }
    ArrayList<Double> cost = computeConfigCostFromList(l);
    int m = cost.size();
    if (number_of_objectives == 0) {
      number_of_objectives = m;
      infinity_cost = new ArrayList<Double>();
      for (int j=0; j<m; j++)
        infinity_cost.add(infinity);
    } else if (m != number_of_objectives)
      throw new IllegalStateException("Inconsistent number of objectives");
    return cost;
  }

  public ArrayList<Double> configCost(ParameterConfiguration config) {
    return configCost(config, getNbRunsFor(config));
  }

  public ArrayList<Double> configCost(ParameterConfiguration config, int k) {
    if (k == 0)
      return infinity_cost;
    // update cache
    updateCache(config, k);
    // compute cost
    int n = getNbRunsFor(config);
    if (n < k)
      return infinity_cost;
    return computeConfigCost(config, k);
  }

  public Boolean updateIncumbents() {
    return updateIncumbents(-1);
  }

  public Boolean updateIncumbents(int nmax) {
    if (incumbentArchive.size() <= 1)
      return false;
    ArrayList<ParameterConfiguration> tmp = new ArrayList<>(incumbentArchive);
    Boolean updated, shrinked;
    int n, nmin = -1;
    for (ParameterConfiguration inc : tmp) {
      n = getNbRunsFor(inc);
      if (nmin == -1)
        nmin = n;
      else
        nmin = Math.min(nmin, n);
      nmax = Math.max(nmax, n);
    }
    updated = (nmin < nmax);
    if (updated)
      ensurePopDetail(incumbentArchive, nmax);
    shrinked = cleanArchive(incumbentArchive);
    if (!updated && !shrinked)
      return false;
    if (updated)
      log.info("Incumbent archive updated!");
    else
      log.info("Incumbent archive modified!");
    for (ParameterConfiguration inc : tmp) {
      if (!isIncumbent(inc)) {
        log.info("No more incumbent: {}", configSIDC(inc));
        fireEvent(new IncumbentPerformanceChangeEvent(termCond, infinity_cost, inc, -1, inc, cpuTime));
      } else if (updated) {
        // TODO
        //log.info("Config {} (possibly) updated!", configSID(inc));
        //fireEvent(new IncumbentPerformanceChangeEvent(termCond, computeConfigCost(inc), inc, nmax, inc, cpuTime));
      }
    }
    printSortedPop(incumbentArchive);
    return true;
  }

  public void checkIncumbent(ParameterConfiguration config) {
    Boolean updated = updateIncumbents(getNbRunsFor(config));
    int nconfig = getNbRunsFor(config); // not necessarily the same
    int nmax = 0;
    for (ParameterConfiguration inc : incumbentArchive)
      nmax = Math.max(nmax, getNbRunsFor(inc));
    if (isIncumbent(config) || getNbRunsFor(config) < nmax)
      return;
    ArrayList<ParameterConfiguration> tmp = new ArrayList<>(incumbentArchive);
    updateArchive(incumbentArchive, config);
    if (!isIncumbent(config))
      return;
    log.info("New incumbent! {}", configSIDC(config));
    fireEvent(new IncumbentPerformanceChangeEvent(termCond, computeConfigCost(config), config, nconfig, config, cpuTime));
    for (ParameterConfiguration inc : tmp) {
      if (!isIncumbent(inc)) {
        log.info("No more incumbent: {}", configSIDC(inc));
        fireEvent(new IncumbentPerformanceChangeEvent(termCond, infinity_cost, inc, -1, inc, cpuTime));
      }
    }
    printSortedPop(incumbentArchive);
  }

  public String formatCost(ArrayList<Double> cost) {
    StringBuilder sb = new StringBuilder();
    for (int i=0; i<cost.size(); i++) {
      if (i == 0)
        sb.append("[");
      else
        sb.append(", ");
      sb.append(formatCost(cost.get(i)));
    }
    sb.append("]");
    return sb.toString();
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
      currentArchive = new ArrayList<>();
      incumbentArchive = new ArrayList<>();
      beforeMainLoop(initialIncumbentList);
      while (!termCond.haveToStop()) {
        iteration++;
        log.info("== Starting Iteration {} ==", iteration);
        log.info("Tuner time: {} elapsed", getTunerTimeS());
        insideMainLoop(iteration);
      }
    } catch (OutOfTimeException e) {
    }
    cleanArchive(currentArchive);
    updateArchive(incumbentArchive, currentArchive);
    if (incumbentArchive.size() == 0)
      incumbentArchive.add(initialIncumbent);
    if (terminationReason == null)
      terminationReason = termCond.getTerminationReason();
    log.info("MOParamILS completed");
  }

  abstract void beforeMainLoop(List<ParameterConfiguration> init);

  abstract void insideMainLoop(int iteration);

  /* ======================================================================
   * Subfunctions
   * ====================================================================== */

  protected ArrayList<ParameterConfiguration> localSearch(ArrayList<ParameterConfiguration> pop) {
    ArrayList<ParameterConfiguration> archive = new ArrayList<>();
    ArrayList<ParameterConfiguration> selected = new ArrayList<>();
    ArrayList<ParameterConfiguration> candidates = new ArrayList<>();
    List<ParameterConfiguration> tabuList = new ArrayList<>();
    // Init
    tabuList.addAll(pop);
    ensurePopDetail(pop, options.minRuns);
    updateArchive(archive, pop);
    while (true) {
      // Cleaning
      log.info("-- cleaning --");
      selected.clear();
      candidates.clear();
      // Selection
      log.info("-- selection --");
      selected.addAll(archive);
      //printPop(selected);
      for (ParameterConfiguration config : selected) {
        // Exploration
        log.info("-- exploration --");
        candidates.addAll(exploration(config, tabuList));
      }
      //printPop(candidates);
      // Archive
      log.info("-- archive --");
      updateArchive(archive, candidates);
      //printPop(archive);
      // in case of termination
      //pop.clear();
      //pop.addAll(archive);
      if (archive.size() == 0)
        throw new IllegalStateException("SHOULD ABSOLUTLY NEVER HAPPEN");
      if (candidates.size() == 0)
        break;
    }
    log.info("-- local search completed --");
    return archive;
  }

  public ArrayList<ParameterConfiguration> getNeighborhood(ParameterConfiguration config) {
    Random configRandN = pool.getRandom("PARAMILS_RANDOM_NEIGHBOR_CONT");
    ArrayList<ParameterConfiguration> pop = new ArrayList<>();
    ArrayList<ParameterConfiguration> list = new ArrayList<>();
    switch (options.scenarioConfig.getRunObjective()) {
    case RUNTIME:
      list.addAll(config.getNeighbourhood(configRandN, options.scenarioConfig.algoExecOptions.paramFileDelegate.continuousNeighbours));
      pop.addAll(list);
      List<String> runtimes = new ArrayList<>(configSpace.getValuesMap().get("runtime"));
      runtimes.remove(config.get("runtime"));
      for (ParameterConfiguration neighbor : list) {
        if (neighbor.get("runtime") != config.get("runtime"))
          continue;
        for (String runtime : runtimes) {
          ParameterConfiguration tmp = new ParameterConfiguration(neighbor);
          tmp.put("runtime", runtime);
          pop.add(tmp);
        }
      }
      break;
    case QUALITY:
      pop.addAll(config.getNeighbourhood(configRandN, options.scenarioConfig.algoExecOptions.paramFileDelegate.continuousNeighbours));
      break;
    }
    return pop;
  }

  // this function moves all configs in nbh that are different from config on the tradeoffParams to the fromt of nbh
  public void prioritizeTradeoffParameters(ParameterConfiguration config, ArrayList<ParameterConfiguration> nbh) {
    for (int i = 0, j = nbh.size()-1; i < j; i++) {
      Boolean priority = false;
      for (String param : options.tradeoffParams) {
        if (nbh.get(i).get(param) != config.get(param)) {
          // priority
          priority = true;
          break;
        }
      }
      if (!priority) {
        Collections.swap(nbh, i, j);
        i--;
        j--;
      }
    }
  }

  protected ArrayList<ParameterConfiguration> exploration(ParameterConfiguration config,
                                                          List<ParameterConfiguration> tabuList) {
    Random configRandLS = pool.getRandom("PARAMILS_LOCAL_SEARCH_NEIGHBOURS");
    ArrayList<ParameterConfiguration> pop = new ArrayList<>();
    ArrayList<ParameterConfiguration> nbh = getNeighborhood(config);
    Collections.shuffle(nbh, configRandLS);
    prioritizeTradeoffParameters(config, nbh);
    if (options.nLocalSearch > 0 && options.nLocalSearch < nbh.size()) {
      nbh = new ArrayList(nbh.subList(0, options.nLocalSearch));
    }
    for (ParameterConfiguration child : nbh) {
        logConfig("trying ", child);
      if (tabuList.contains(child)) {
        if (options.logDetails)
          log.info(" ... tabu");
      } else {
        checkTunerTime();
        tabuList.add(child);
        switch (options.scenarioConfig.getRunObjective()) {
        case RUNTIME:
          List<String> runtimes = configSpace.getValuesMap().get("runtime");
          Boolean pruned = false;
          for (String runtime : runtimes) {
            if (pruned)
              continue;
            if (runtime == child.get("runtime"))
              continue;
            if (Double.parseDouble(runtime) < Double.parseDouble(child.get("runtime")))
              continue;
            ParameterConfiguration childn = new ParameterConfiguration(child);
            childn.put("runtime", runtime);
            if (getHistoryHash().get(childn) == null)
              continue;
            if (!dominates(config, childn))
              continue;
            pruned = true;
          }
          if (pruned)
            continue;
          break;
        }
        if (better(child, config)) {
          if (options.logDetails) {
            int k1 = getNbRunsFor(child);
            int k2 = getNbRunsFor(config);
            log.info(" ... new local best!");
            log.info(" {} -> {} (based on {} and {} runs)", formatCost(computeConfigCost(config, k2)), formatCost(computeConfigCost(child, k1)), k2, k1);
          }
          pop.add(child);
          break;
        } else if (!better(config, child)) {
          if (options.logDetails) {
            int k1 = getNbRunsFor(child);
            int k2 = getNbRunsFor(config);
            log.info(" ... new neutral local best!");
            log.info(" {} -> {} (based on {} and {} runs)", formatCost(computeConfigCost(config, k2)), formatCost(computeConfigCost(child, k1)), k2, k1);
          }
          pop.add(child);
        } else {
          if (options.logDetails) {
            int k1 = getNbRunsFor(child);
            int k2 = getNbRunsFor(config);
            if (k1 < k2) {
              log.info(" ... capped after {}/{} runs", k1, k2);
            } else {
              log.info(" ... worse");
            }
          }
        }
        checkIncumbent(child);
      }
    }
    return pop;
  }

  abstract Boolean better(ParameterConfiguration a, ParameterConfiguration b);

  protected Boolean dominates(ParameterConfiguration a,
                              ParameterConfiguration b) {
    int nA = getNbRunsFor(a);
    int nB = getNbRunsFor(b);
    if (nA < nB) {
      return false;
    } else {
      return paretoSB(a, b, nB);
    }
  }

  protected Boolean dominatesOrEquiv(ParameterConfiguration a,
                                     ParameterConfiguration b) {
    int nA = getNbRunsFor(a);
    int nB = getNbRunsFor(b);
    if (nA < nB) {
      return false;
    } else {
      return paretoBE(a, b, nB);
    }
  }

  protected Boolean equiv(ParameterConfiguration a,
                          ParameterConfiguration b) {
    int nA = getNbRunsFor(a);
    int nB = getNbRunsFor(b);
    if (nA != nB) {
      return false;
    } else {
      return paretoBE(a, b, nB) && !paretoSB(a, b, nB);
    }
  }

  /* ======================================================================
   * Helper functions
   * ====================================================================== */

  public Boolean paretoBE(ParameterConfiguration a,
                          ParameterConfiguration b) {
    int n = Math.min(getNbRunsFor(a), getNbRunsFor(b));
    return paretoBE(a, b, n);
  }

  public Boolean paretoBE(ParameterConfiguration a,
                          ParameterConfiguration b,
                          int n) {
    return paretoBE(computeConfigCost(a, n), computeConfigCost(b, n));
  }

  public Boolean paretoBE(ArrayList<Double> a,
                          ArrayList<Double> b) {
    return !paretoSB(b, a);
  }

  public Boolean paretoSB(ParameterConfiguration a,
                          ParameterConfiguration b) {
    int n = Math.min(getNbRunsFor(a), getNbRunsFor(b));
    return paretoSB(a, b, n);
  }

  public Boolean paretoSB(ParameterConfiguration a,
                          ParameterConfiguration b,
                          int n) {
    return paretoSB(computeConfigCost(a, n), computeConfigCost(b, n));
  }

  public Boolean paretoSB(ArrayList<Double> a,
                          ArrayList<Double> b) {
    int m = a.size();
    if (m != b.size())
      throw new IllegalStateException("Insuffisant detail");
    for (int k=0; k<m; k++)
      if (a.get(k) > b.get(k))
        return false;
    return true;
  }

  protected ParameterConfiguration randomArchiveMember(ArrayList<ParameterConfiguration> archive) {
    Random configRandM = pool.getRandom("PARAMILS_RANDOM_ARCHIVE_MEMBER");
    Collections.shuffle(archive, configRandM);
    return archive.get(0);
  }

  public Boolean updateArchive(ArrayList<ParameterConfiguration> archive,
                               ArrayList<ParameterConfiguration> configs) {
    Boolean updated = false;
    for (ParameterConfiguration config : configs)
      if (updateArchive(archive, config))
        updated = true;
    return updated;
  }

  public Boolean updateArchive(ArrayList<ParameterConfiguration> archive,
                               ParameterConfiguration config) {
    if (getNbRunsFor(config) == 0)
      return false;
    if (!archive.contains(config))
      archive.add(config);
    cleanArchive(archive);
    return archive.contains(config);
  }

  public Boolean cleanArchive(ArrayList<ParameterConfiguration> archive) {
    ArrayList<ParameterConfiguration> cleaned = new ArrayList<>();
    ArrayList<ParameterConfiguration> toRemove = new ArrayList<>();
    Boolean shrinked = false;
    Boolean dominated;
    for (ParameterConfiguration config : archive) {
      dominated = false;
      toRemove.clear();
      for (ParameterConfiguration current : cleaned) {
        if (dominated)
          break;
        Boolean d1 = dominates(config, current);
        Boolean d2 = dominates(current, config);
        // TODO: = -> IGNORE, REPLACE, ADD // RANDOM?
        int c = 1;
        if (d1 && d2) { // f(config) = f(current)
          if (c == 1)
            toRemove.add(current);
          if (c == 0)
            dominated = true;
        } else {
          if (d1)
            toRemove.add(current);
          if (d2)
            dominated = true;
        }
      }
      cleaned.removeAll(toRemove);
      if (dominated)
        shrinked = true;
      else
        cleaned.add(config);
    }
    if (cleaned.size() == 0 && archive.size() > 0)
      throw new IllegalStateException("WTF");
    archive.clear();
    archive.addAll(cleaned);
    return shrinked;
  }

  public void ensurePopDetail(ArrayList<ParameterConfiguration> pop, int n) {
    for (ParameterConfiguration config : pop)
      ensureConfigDetail(config, n);
  }

  public void printSortedPop(ArrayList<ParameterConfiguration> pop) {
    ArrayList<ParameterConfiguration> sorted = new ArrayList<>();
    HashMap<ParameterConfiguration, Double> h = new HashMap<>();
    ArrayList<Double> vsorted = new ArrayList<>();
    Double v;
    for (ParameterConfiguration config : pop) {
      v = computeConfigCost(config).get(0);
      vsorted.add(v);
      h.put(config, v);
    }
    Collections.sort(vsorted);
    for (Double x : vsorted)
      for (ParameterConfiguration config : pop)
        if (x == h.get(config)) {
          sorted.add(config);
          h.remove(config);
        }
    printPop(sorted);
  }

  public void printPop(ArrayList<ParameterConfiguration> pop) {
    log.info("size: {}", pop.size());
    for (ParameterConfiguration tmp : pop)
      printConfig(tmp);
  }

  public void printConfig(ParameterConfiguration config) {
    int n = getNbRunsFor(config);
    if (n > 0)
      printConfigCost(config, n, computeConfigCost(config));
    else
      printConfigCost(config, 0, infinity_cost);
  }

  public void printConfigCost(ParameterConfiguration config, int n, ArrayList<Double> cost) {
    log.info(" {} with {} run{}: {}", configSIDC(config), n, (n > 1 ? "s" : ""), configSP(config));
    if (n > 0)
      log.info("  -> " + formatCost(cost));
  }

  public void printConfigCallString(ParameterConfiguration config) {
    int n = getNbRunsFor(config);
    log.info("Callstring for {}: {}\n{}", configSIDC(config), configSP(config), getCallString(config));
  }

  public Boolean isIncumbent(ParameterConfiguration config) {
    return belongsTo(config, incumbentArchive);
  }

  public Boolean belongsTo(ParameterConfiguration config,
                           ArrayList<ParameterConfiguration> pop) {
    for (ParameterConfiguration ref : pop)
      if (config.equals(ref))
        return true;
    return false;
  }

  /* ======================================================================
   * Final Report
   * ====================================================================== */

  public void logFinalReport() {
    final DecimalFormat df0 = new DecimalFormat("0");
    log.info("Total number of runs performed: {} ({}), total configurations tried: {}", statNbRuns, statNbUniqRuns, getTotalConfs());
    log.info("Total CPU time used: {} s, total wallclock time used: {} s", df0.format(termCond.getTunerTime()), df0.format(termCond.getWallTime()));
    log.info("Final archive:");
    printSortedPop(incumbentArchive);
    if (!options.doValidation) {
      for (ParameterConfiguration config : incumbentArchive)
        printConfigCallString(config);
    }
    for (ParameterConfiguration config : incumbentArchive)
      logConfigOnFile(config);
  }

  /* ======================================================================
   * Validation
   * ====================================================================== */

  public void doValidation() {
    log.info("Testing final incumbents");
    validationArchive = new ArrayList<>();
    validationArchive.addAll(incumbentArchive);
    for (ParameterConfiguration config : validationArchive) {
      updateCache(config, options.validationRuns);
      printConfig(config);
    }
    cleanArchive(validationArchive);
  }

  /* ======================================================================
   * Final Validation Report
   * ====================================================================== */

  public void logValidationMessage() {
    log.info("Estimated {} of final incumbent archive on test set:", objectiveToReport);
    int n = testBenchmark.size();
    printSortedPop(validationArchive);
    log.info("Estimations based on {} run{} on {} test instance{}", testBenchmark.size(), (testBenchmark.size() > 1 ? "s" : ""), testInstances.size(), (testInstances.size() > 1 ? "s" : ""));
    log.info("------------------------------------------------------------------------");
    for (ParameterConfiguration config : validationArchive)
      printConfigCallString(config);
  }
}
