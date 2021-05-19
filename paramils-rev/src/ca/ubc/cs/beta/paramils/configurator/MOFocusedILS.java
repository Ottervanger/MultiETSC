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
import ca.ubc.cs.beta.paramils.misc.ExecutionMode;

public class MOFocusedILS extends MOParamILS {

  /* ======================================================================
   * Constructor
   * ====================================================================== */

  public MOFocusedILS(ParamILSOptions paramilsOptions,
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
    super(paramilsOptions, execConfig, instances, algoEval, configSpace, instanceSeedGen, initialIncumbentList, manager, rh, pool, termCond, cpuTime);
  }

  /* ======================================================================
   * Main algorithm
   * ====================================================================== */

  protected void beforeMainLoop(List<ParameterConfiguration> initList) {
    currentArchive.addAll(initList);
    ensurePopDetail(currentArchive, options.minRuns);
    cleanArchive(currentArchive);
    for (ParameterConfiguration config : currentArchive)
      checkIncumbent(config);
    if (options.R > 0)
      log.info("Looking for random initial solutions better than the initial incumbent");
    for (int i=0; i<options.R; i++) {
      ParameterConfiguration random = randomConfig();
      logConfig("trying ", random);
      if (compareToPop(random, currentArchive))
        updateArchive(currentArchive, random);
    }
    switch(options.bonusRunsMode) {
    case INTENSIFY:
      log.info("Intensifying...");
      for (ParameterConfiguration config : currentArchive)
        intensify(config);
      cleanArchive(currentArchive);
      break;
    case LEGACY:
      bonusRuns = 0;
      break;
    }
  }

  protected void insideMainLoop(int iteration) {
    ArrayList<ParameterConfiguration> tmpPop = new ArrayList<>();
    // perturbation
    if (iteration > 1) {
      ParameterConfiguration tmp;
      Random probaRandom = pool.getRandom("PARAMILS_RANDOM_RESTART");
      if (probaRandom.nextFloat() < options.randomRestartProbability) {
        log.info("Restarting");
        tmp = randomConfig();
        currentArchive.clear();
        currentArchive.add(tmp);
      } else {
        log.info("Perturbing");
        tmp = randomArchiveMember(currentArchive);
        for (int i=0; i<options.perturbationLength; i++)
          tmp = randomNeighbor(tmp);
      }
      tmpPop.add(tmp);
      // pre local search
      ensurePopDetail(tmpPop, options.minRuns);
      switch (options.bonusRunsMode) {
      case INTENSIFY:
        for (ParameterConfiguration config : tmpPop)
          intensify(config);
        break;
      }
    } else {
      tmpPop.addAll(currentArchive);
    }
    // local search
    log.info("Initial archive:");
    printSortedPop(tmpPop);
    tmpPop = localSearch(tmpPop);
    // post local search
    switch (options.bonusRunsMode) {
    case LEGACY:
      int indvruns = bonusRuns / tmpPop.size();
      for (ParameterConfiguration config : tmpPop) {
        bonusRuns = indvruns;
        useBonusRunsOn(config);
      }
      break;
    case INTENSIFY:
      for (ParameterConfiguration config : tmpPop)
        intensify(config);
      break;
    }
    // acceptance criterion
    for (ParameterConfiguration config : tmpPop)
      compareToPop(config, currentArchive);
    cleanArchive(tmpPop);
    updateArchive(currentArchive, tmpPop);
    // ensure incumbent comparison
    for (ParameterConfiguration config : currentArchive)
      compareToPop(config, incumbentArchive);
    cleanArchive(currentArchive);
    updateIncumbents();
    // log
    log.info("Local optimas:");
    printSortedPop(tmpPop);
    log.info("Local incumbent archive:");
    printSortedPop(currentArchive);
    log.info("Global incumbent archive:");
    printSortedPop(incumbentArchive);
  }

  /* ======================================================================
   * Subfunctions
   * ====================================================================== */

  protected Boolean better(ParameterConfiguration a, ParameterConfiguration b) {
    int nA, nB;
    Boolean is_better;
    // before comparison
    switch(options.bonusRunsMode) {
    case LEGACY:
      if (getNbRunsFor(a) == getNbRunsFor(b))
        bonusRuns += 2;
      else
        bonusRuns += 1;
    }
    // comparison
    ensureConfigDetail(a, options.minRuns);
    ensureConfigDetail(b, options.minRuns);
    do {
      nA = getNbRunsFor(a);
      nB = getNbRunsFor(b);
      if (nA < nB)
        ensureConfigDetail(a, nA+1);
      else if (nB < nA)
        ensureConfigDetail(b, nB+1);
      else // nA == nB
        break;
    } while (!dominates(a, b) && !dominates(b, a));
    is_better = dominates(a, b);
    // after comparison
    switch(options.bonusRunsMode) {
    case LEGACY:
      if (is_better && getNbRunsFor(a) == getNbRunsFor(b))
        useBonusRunsOn(a);
      break;
    case INTENSIFY:
      if (is_better && getNbRunsFor(a) == getNbRunsFor(b))
        intensify(a);
      break;
    }
    // ensure incumbent detail
    checkIncumbent(a);
    checkIncumbent(b);
    // final comparison
    return is_better;
  }

  protected Boolean compareToPop(ParameterConfiguration config,
                                 ArrayList<ParameterConfiguration> pop) {
    int nA, nB;
    Boolean is_better;
    // before comparison
    nA = getNbRunsFor(config);
    nB = options.maxRuns;
    for (ParameterConfiguration ref : pop)
      if (config.equals(ref))
        return true;
      else
        nB = Math.min(nB, getNbRunsFor(ref));
    switch(options.bonusRunsMode) {
    case LEGACY:
      if (nA == nB)
        bonusRuns += 2;
      else
        bonusRuns += 1;
    }
    // comparison
    ensureConfigDetail(config, options.minRuns);
    ensurePopDetail(pop, options.minRuns);
    Boolean do_break;
    do {
      nA = getNbRunsFor(config);
      if (nA >= nB)
        break;
      ensureConfigDetail(config, nA+1);
      for (ParameterConfiguration ref : pop)
        if (dominates(ref, config))
          return false;
      do_break = false;
      for (ParameterConfiguration ref : pop) {
        if (dominates(config, ref) || dominates(ref, config)) {
          do_break = true;
          break;
        }
      }
      if (do_break)
        break;
    } while (true);
    // after comparison
    is_better = false;
    for (ParameterConfiguration ref : pop)
      if (dominates(config, ref))
        is_better = true;
    switch(options.bonusRunsMode) {
    case LEGACY:
      if (is_better)
        useBonusRunsOn(config);
      break;
    case INTENSIFY:
      if (is_better)
        intensify(config);
      break;
    }
    // ensure incumbent detail
    checkIncumbent(config);
    return true;
  }

  protected void useBonusRunsOn(ParameterConfiguration config) {
    if (options.logDetails)
      log.info(bonusRuns + " bonus run" + (bonusRuns > 1 ? "s" : "") + " on " + configSID(config));
    ensureConfigDetail(config, getNbRunsFor(config) + bonusRuns);
    // ensure incumbent detail
    checkIncumbent(config);
    bonusRuns = 0;
  }

  protected int intensify(ParameterConfiguration config) {
    int bonus = 0;
    int n;
    ArrayList<Double> cost = null;
    ArrayList<Double> tmp = null;
    if (options.logDetails)
      log.info(" ... intensifying " + configSID(config));
    do {
      n = getNbRunsFor(config);
      if (n == options.maxRuns)
        break;
      if (bonus == 0)
        tmp = configCost(config, n);
      else
        tmp = cost;
      cost = configCost(config, n+1);
      bonus += 1;
    } while (paretoBE(tmp, cost));
    if (options.logDetails && bonus > 0)
      log.info(bonus + " bonus run" + (bonus > 1 ? "s" : "") + " on " + configSID(config));
    // ensure incumbent detail
    checkIncumbent(config);
    return bonus;
  }
}
