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

public class SOBasicILS extends SOParamILS {

  /* ======================================================================
   * Constructor
   * ====================================================================== */

  public SOBasicILS(ParamILSOptions paramilsOptions,
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
    logConfig("Initial incumbent: ", init);
    current = init;
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
          if (k1 < k2)
            log.info(" ... capped after {}/{} runs", k1, k2);
          else
            log.info(" ... worse");
        }
      }
    }
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
    } else {
      tmp = current;
    }
    ensureConfigDetail(tmp, options.minRuns);
    int k = getNbRunsFor(tmp);
    log.info("Initial configuration ({})", configSID(tmp));
    log.info(" -> {} (based on {} run{})", formatCost(computeConfigCost(tmp, k)), k, (k > 1 ? "s" : ""));
    // local search
    tmp = localSearch(tmp);
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
    int n = options.maxRuns;
    switch (options.cappingMode) {
    case OFF:
      ensureConfigDetail(b, n);
      ensureConfigDetail(a, n);
      break;
    case BASIC:
    case FINE:
      double cost, budget, ibudget, bm;
      cost = configCost(b, n);
      budget = cost*n;
      if (options.aggressiveCapping) {
        bm = options.aggressiveCappingFactor; // bound multiplier
        ibudget = configCost(incumbent, n)*bm*n;
        budget = Math.min(budget, ibudget);
      }
      configCostWithBudget(a, n, budget);
      break;
    default:
      throw new IllegalStateException("Not sure what to default to");
    }
    // ensure incumbent comparison
    checkIncumbent(a);
    checkIncumbent(b);
    // compare
    return dominates(a, b);
  }
}
