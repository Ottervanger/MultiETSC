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

public class SOFocusedILS extends SOParamILS {

  /* ======================================================================
   * Constructor
   * ====================================================================== */

  public SOFocusedILS(ParamILSOptions paramilsOptions,
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
  }

  /* ======================================================================
   * Main algorithm
   * ====================================================================== */

  protected void beforeMainLoop(ParameterConfiguration init) {
    // setup initial incumbent
    bonusRuns = 0;
    logConfig("Initial incumbent: ", init);
    current = init;
    switch (options.bonusRunsMode) {
    case INTENSIFY:
      intensify(init);
      break;
    }
    // random encounters
    log.info("Looking for random initial solutions better than the initial incumbent");
    for (int k=0; k<options.R; k++) {
      ParameterConfiguration random = randomConfig();
      logConfig("trying ", random);
      if (better(random, current)) {
        int k1 = getNbRunsFor(random);
        int k2 = getNbRunsFor(current);
        if (options.logDetails) {
          log.info(" ... better!");
          log.info(" {} -> {} (based on {} and {} runs)", formatCost(computeConfigCost(current, k2)), formatCost(computeConfigCost(random, k1)), k2, k1);
        }
        current = random;
        checkIncumbent(current);
      } else {
        int k1 = getNbRunsFor(random);
        int k2 = getNbRunsFor(current);
        if (options.logDetails) {
          if (k1 < k2) {
            log.info(" ... capped after {}/{} runs", k1, k2);
          } else {
            log.info(" ... worse");
          }
        }
      }
    }
    current = incumbent;
  }

  protected void insideMainLoop(int iteration) {
    // perturbation
    if (iteration > 1) {
      Random probaRandom = pool.getRandom("PARAMILS_RANDOM_RESTART");
      if (probaRandom.nextFloat() < options.randomRestartProbability) {
        random = randomConfig();
        log.info("Restarting");
        log.info(" {}", reportDiff(current, random, " ; "));
        current = random;
        tmp = random;
      } else {
        log.info("Perturbing");
        tmp = current;
        for (int i=0; i<options.perturbationLength; i++)
          tmp = randomNeighbor(tmp);
        log.info(" {}", reportDiff(current, tmp, " ; "));
      }
      // pre local search
      ensureConfigDetail(tmp, options.minRuns);
      switch(options.bonusRunsMode) {
      case INTENSIFY:
        intensify(tmp);
        break;
      }
    } else {
      tmp = current;
    }
    int k = getNbRunsFor(tmp);
    log.info("Initial configuration ({})", configSID(tmp));
    log.info(" -> {} (based on {} run{})", formatCost(computeConfigCost(tmp, k)), k, (k > 1 ? "s" : ""));
    // local search
    tmp = localSearch(tmp);
    // post local search
    switch(options.bonusRunsMode) {
    case LEGACY:
      useBonusRunsOn(tmp);
      break;
    case INTENSIFY:
      intensify(tmp);
      break;
    }
    // acceptance criterion
    if (better(tmp, current)) {
      int k1 = getNbRunsFor(tmp);
      int k2 = getNbRunsFor(current);
      log.info("Local optima " + configSID(tmp) + " accepted");
      log.info(" {} -> {} (based on {} and {} runs)", formatCost(computeConfigCost(current, k2)), formatCost(computeConfigCost(tmp, k1)), k2, k1);
      current = tmp;
    } else {
      int k1 = getNbRunsFor(tmp);
      int k2 = getNbRunsFor(current);
      log.info("Local optima " + configSID(tmp) + " discarded");
      log.info(" {} -/-> {} (based on {} and {} runs)", formatCost(computeConfigCost(current, k2)), formatCost(computeConfigCost(tmp, k1)), k2, k1);
    }
    // ensure incumbent comparison
    if (current != incumbent)
      better(current, incumbent);
  }

  /* ======================================================================
   * Subfunctions
   * ====================================================================== */

  protected Boolean better(ParameterConfiguration a, ParameterConfiguration b) {
    double budget, ibudget;
    double bm = options.aggressiveCappingFactor; // bound multiplier
    int nA, nB, n_budget;
    Boolean is_better;
    // before comparison
    switch(options.bonusRunsMode) {
    case LEGACY:
      if (getNbRunsFor(a) == getNbRunsFor(b))
        bonusRuns += 2;
      else
        bonusRuns += 1;
      break;
    }
    // finish partial runs
    if (configWasCapped(a))
      ensureConfigDetail(a, getNbRunsFor(a));
    if (configWasCapped(b))
      ensureConfigDetail(b, getNbRunsFor(b));
    // comparison
    switch (options.cappingMode) {
    case OFF:
      ensureConfigDetail(a, options.minRuns);
      ensureConfigDetail(b, options.minRuns);
      do {
        nA = getNbRunsFor(a);
        nB = getNbRunsFor(b);
        if (nA < nB) {
          ensureConfigDetail(a, nA+1);
        } else if (nB < nA) {
          ensureConfigDetail(b, nB+1);
        } else { // nA == nB
          if (nA == options.maxRuns)
            break;
          ensureConfigDetail(a, nA+1);
          ensureConfigDetail(b, nB+1);
        }
      } while (!dominates(a, b) && !dominates(b, a));
      break;
    case BASIC:
    case FINE:
      double cost;
      do {
        nA = getNbRunsFor(a);
        nB = getNbRunsFor(b);
        if (nA < nB) {
          n_budget = Math.max(nA+1, options.minRuns);
          budget = configCost(b, nA+1)*n_budget;
          if (options.aggressiveCapping) {
            ibudget = cost_incumbent*bm*n_budget;
            budget = Math.min(budget, ibudget);
          }
          cost = configCostWithBudget(a, nA+1, budget);
          if (cost == infinity)
            break; // capped
        } else if (nB < nA) {
          n_budget = Math.max(nB+1, options.minRuns);
          budget = configCost(a, nB+1)*n_budget;
          if (options.aggressiveCapping) {
            ibudget = cost_incumbent*bm*n_budget;
            budget = Math.min(budget, ibudget);
          }
          configCostWithBudget(b, nB+1, budget);
          cost = configCostWithBudget(b, nB+1, budget);
          if (cost == infinity)
            break; // capped
        } else { // nA == nB
          if (nA == options.maxRuns)
            break;
          Boolean dobreak = false;
          switch (options.bonusRunsMode) {
          case INTENSIFY:
            intensify(a);
            // intensify(b); // why not?
            break;
          default:
            n_budget = Math.max(nB+1, options.minRuns);
            budget = configCost(b, nB+1)*n_budget;
            if (options.aggressiveCapping) {
              ibudget = cost_incumbent*bm*n_budget;
              budget = Math.min(budget, ibudget);
            }
            cost = configCostWithBudget(a, nB+1, budget);
            if (cost == infinity)
              dobreak = true; // capped
          }
          if (dobreak)
            break;
        }
      } while (!dominates(a, b) && !dominates(b, a));
      break;
    default:
      throw new IllegalStateException("Not sure what to default to");
    }
    is_better = dominates(a, b);
    // after comparison
    switch(options.bonusRunsMode) {
    case LEGACY:
      if (is_better)
        useBonusRunsOn(a);
      break;
    case INTENSIFY:
      if (is_better)
        intensify(a);
      else if (getNbRunsFor(a) == getNbRunsFor(b))
        intensify(b);
      break;
    }
    // check incumbent
    checkIncumbent(a);
    checkIncumbent(b);
    // final comparison
    return is_better;
  }

  protected void useBonusRunsOn(ParameterConfiguration config) {
    log.info("Spending " + bonusRuns + " bonus run" + (bonusRuns > 1 ? "s" : "") + " on " + configSID(config));
    ensureConfigDetail(config, getNbRunsFor(config) + bonusRuns);
    // check incumbent
    checkIncumbent(config);
    bonusRuns = 0;
  }

  protected int intensify(ParameterConfiguration config) {
    int bonus = 0;
    int n;
    int n_i = getNbRunsFor(incumbent);
    double cost = 0;
    double tmp = 0;
    if (options.logDetails)
      log.info("Intensifying {} ...", configSID(config));
    // intensification
    do {
      n = getNbRunsFor(config);
      if (n == options.maxRuns)
        break;
      if (bonus == 0) {
        tmp = configCost(config, n);
        if (n > 0 && tmp == infinity)
          return 0;
        // config was not capped
      } else {
        tmp = cost;
      }
      cost = configCost(config, n+1);
      if (cost == infinity)
        break;
      bonus += 1;
    } while (tmp < cost);
    if (options.logDetails)
      log.info(" ... {} bonus run{} on {}", bonus, (bonus > 1 ? "s" : ""), configSID(config));
    else if (bonus > 0)
      log.info("Intensifying {} ... {}  bonus run{}", configSID(config), bonus, (bonus > 1 ? "s" : ""));
    // check incumbent
    checkIncumbent(config);
    return bonus;
  }
}
