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

package ca.ubc.cs.beta.paramils.builder;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.BufferedReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.beust.jcommander.ParameterException;

import ca.ubc.cs.beta.aeatk.acquisitionfunctions.AcquisitionFunctions;
import ca.ubc.cs.beta.aeatk.algorithmexecutionconfiguration.AlgorithmExecutionConfiguration;
import ca.ubc.cs.beta.aeatk.eventsystem.EventManager;
import ca.ubc.cs.beta.aeatk.eventsystem.events.ac.AutomaticConfigurationEnd;
import ca.ubc.cs.beta.aeatk.eventsystem.events.ac.IncumbentPerformanceChangeEvent;
import ca.ubc.cs.beta.aeatk.misc.cputime.CPUTime;
import ca.ubc.cs.beta.aeatk.objectives.OverallObjective;
import ca.ubc.cs.beta.aeatk.objectives.RunObjective;
import ca.ubc.cs.beta.aeatk.options.AbstractOptions;
import ca.ubc.cs.beta.aeatk.options.scenario.ScenarioOptions;
import ca.ubc.cs.beta.aeatk.parameterconfigurationspace.ParameterConfiguration;
import ca.ubc.cs.beta.aeatk.parameterconfigurationspace.ParameterConfigurationSpace;
import ca.ubc.cs.beta.aeatk.parameterconfigurationspace.ParameterConfiguration.ParameterStringFormat;
import ca.ubc.cs.beta.aeatk.probleminstance.InstanceListWithSeeds;
import ca.ubc.cs.beta.aeatk.probleminstance.ProblemInstance;
import ca.ubc.cs.beta.aeatk.probleminstance.seedgenerator.InstanceSeedGenerator;
import ca.ubc.cs.beta.aeatk.random.SeedableRandomPool;
import ca.ubc.cs.beta.aeatk.random.SeedableRandomPoolConstants;
import ca.ubc.cs.beta.aeatk.runhistory.FileSharingRunHistoryDecorator;
import ca.ubc.cs.beta.aeatk.runhistory.RunHistory;
import ca.ubc.cs.beta.aeatk.runhistory.ThreadSafeRunHistoryWrapper;
import ca.ubc.cs.beta.aeatk.targetalgorithmevaluator.TargetAlgorithmEvaluator;
import ca.ubc.cs.beta.aeatk.targetalgorithmevaluator.decorators.helpers.TargetAlgorithmEvaluatorNotifyTerminationCondition;
import ca.ubc.cs.beta.aeatk.termination.CompositeTerminationCondition;
import ca.ubc.cs.beta.aeatk.trajectoryfile.TrajectoryFileLogger;

import ca.ubc.cs.beta.paramils.configurator.AbstractAlgorithmFramework;
import ca.ubc.cs.beta.paramils.configurator.SOBasicILS;
import ca.ubc.cs.beta.paramils.configurator.SOFocusedILS;
import ca.ubc.cs.beta.paramils.configurator.SOParamILS;
import ca.ubc.cs.beta.paramils.configurator.MOBasicILS;
import ca.ubc.cs.beta.paramils.configurator.MOFocusedILS;
import ca.ubc.cs.beta.paramils.configurator.MOParamILS;
import ca.ubc.cs.beta.paramils.misc.ApproachMode;
import ca.ubc.cs.beta.paramils.misc.ConfigWriter;
import ca.ubc.cs.beta.paramils.misc.ExecutionMode;
import ca.ubc.cs.beta.paramils.misc.ParamILSOptions;
import ca.ubc.cs.beta.paramils.misc.ParamILSRunHistory;

/**
 * Builds an Automatic Configurator
 */

public class ParamILSBuilder {

  private static transient Logger log = LoggerFactory.getLogger(ParamILSBuilder.class);

  private final EventManager eventManager;

  private volatile TrajectoryFileLogger tLog;

  public ParamILSBuilder() {
    this.eventManager = new EventManager();
  }

  public EventManager getEventManager() {
    return eventManager;
  }

  public AbstractAlgorithmFramework getAutomaticConfigurator(AlgorithmExecutionConfiguration execConfig, InstanceListWithSeeds trainingILWS, ParamILSOptions options,Map<String, AbstractOptions> taeOptions, String outputDir, SeedableRandomPool pool) {
    CPUTime cpuTime = new CPUTime();

    ParameterConfigurationSpace configSpace = execConfig.getParameterConfigurationSpace();

    double configSpaceSize = configSpace.getUpperBoundOnSize();

    if (Double.isInfinite(configSpaceSize)) {
      log.debug("Configuration Space has at least one continuous parameter or is very large (only bound expressible in IEEE 754 format is Infinity)");
    } else {
      log.debug("Configuration Space size is at most {}", configSpace.getUpperBoundOnSize());
    }

    List<ProblemInstance> instances = trainingILWS.getInstances();
    InstanceSeedGenerator instanceSeedGen = trainingILWS.getSeedGen();

    ParameterConfiguration initialIncumbent;

    ArrayList<ParameterConfiguration> initialIncumbentList = new ArrayList<>();

    switch (options.initialIncumbent) {
    case "FILE":
      String path = options.initialIncumbentFile;
      if (path == "")
        throw new IllegalStateException("initialIncumbent is FILE and initialIncumbentFilename is empty");
      try {
        BufferedReader br = new BufferedReader(new FileReader(path));
        String line = null;
        while ((line = br.readLine()) != null)
          initialIncumbentList.add(configSpace.getParameterConfigurationFromString(line, ParameterStringFormat.NODB_SYNTAX, pool.getRandom(SeedableRandomPoolConstants.INITIAL_INCUMBENT_SELECTION)));
        if (initialIncumbentList.size() == 0)
          throw new IllegalStateException("the initialIncumbent file is empty");
        initialIncumbent = initialIncumbentList.get(0);
      } catch (FileNotFoundException e) {
        throw new IllegalStateException("File not found: ``"+path+"''");
      } catch (IOException e) {
        throw new IllegalStateException("IOException while reading: ``"+path+"''");
      }
      break;
    default:
      initialIncumbent = configSpace.getParameterConfigurationFromString(options.initialIncumbent, ParameterStringFormat.NODB_SYNTAX, pool.getRandom(SeedableRandomPoolConstants.INITIAL_INCUMBENT_SELECTION));
      initialIncumbentList.add(initialIncumbent);
      break;
    }

    if (!initialIncumbent.equals(configSpace.getDefaultConfiguration())) {
      log.debug("Initial Incumbent set to \"{}\" ", initialIncumbent.getFormattedParameterString(ParameterStringFormat.NODB_SYNTAX));
    } else {
      log.debug("Initial Incumbent is the default \"{}\" ", initialIncumbent.getFormattedParameterString(ParameterStringFormat.NODB_SYNTAX));
    }

    TargetAlgorithmEvaluator tae = options.scenarioConfig.algoExecOptions.taeOpts.getTargetAlgorithmEvaluator( taeOptions, outputDir, options.seedOptions.numRun);

    RunHistory rh = new ThreadSafeRunHistoryWrapper(new ParamILSRunHistory());

    CompositeTerminationCondition termCond = options.scenarioConfig.limitOptions.getTerminationConditions(cpuTime);

    tLog = new TrajectoryFileLogger(rh, termCond, outputDir + File.separator + "traj-run-" + options.seedOptions.numRun, initialIncumbent, cpuTime);
    eventManager.registerHandler(IncumbentPerformanceChangeEvent.class, tLog);
    eventManager.registerHandler(AutomaticConfigurationEnd.class, tLog);

    termCond.registerWithEventManager(eventManager);

    TargetAlgorithmEvaluator acTae = new TargetAlgorithmEvaluatorNotifyTerminationCondition(tae, eventManager, termCond, true);

    AbstractAlgorithmFramework paramils;
    if (options.multiObjective)
      switch (options.approach) {
      case BASIC:
        paramils = new MOBasicILS(options, execConfig, instances, acTae, configSpace, instanceSeedGen, initialIncumbentList, eventManager, rh, pool, termCond, cpuTime);
        break;
      case FOCUSED:
        paramils = new MOFocusedILS(options, execConfig, instances, acTae, configSpace, instanceSeedGen, initialIncumbentList, eventManager, rh, pool, termCond, cpuTime);
        break;
      default:
        throw new IllegalStateException("Not sure what to default to");
      }
    else
      switch (options.approach) {
      case BASIC:
        paramils = new SOBasicILS(options, execConfig, instances, acTae, configSpace, instanceSeedGen, initialIncumbent, eventManager, rh, pool, termCond, cpuTime);
        break;
      case FOCUSED:
        paramils = new SOFocusedILS(options, execConfig, instances, acTae, configSpace, instanceSeedGen, initialIncumbent, eventManager, rh, pool, termCond, cpuTime);
        break;
      default:
        throw new IllegalStateException("Not sure what to default to");
      }

    ConfigWriter cw = new ConfigWriter(outputDir + File.separator + "configs" + options.seedOptions.numRun);
    paramils.setupConfigWriter(cw);

    return paramils;
  }

  public TrajectoryFileLogger getTrajectoryFileLogger() {
    return tLog;
  }

}
