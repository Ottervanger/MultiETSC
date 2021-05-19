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

package ca.ubc.cs.beta.paramils.executors;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.lang.management.ManagementFactory;
import java.net.InetAddress;
import java.net.UnknownHostException;
import java.text.DecimalFormat;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.SortedMap;
import java.util.TreeMap;
import java.util.Map.Entry;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.slf4j.Marker;
import org.slf4j.MarkerFactory;

import ca.ubc.cs.beta.aeatk.algorithmexecutionconfiguration.AlgorithmExecutionConfiguration;
import ca.ubc.cs.beta.aeatk.exceptions.StateSerializationException;
import ca.ubc.cs.beta.aeatk.exceptions.TrajectoryDivergenceException;
import ca.ubc.cs.beta.aeatk.logging.CommonMarkers;
import ca.ubc.cs.beta.aeatk.misc.jcommander.JCommanderHelper;
import ca.ubc.cs.beta.aeatk.misc.returnvalues.AEATKReturnValues;
import ca.ubc.cs.beta.aeatk.misc.spi.SPIClassLoaderHelper;
import ca.ubc.cs.beta.aeatk.misc.version.JavaVersionInfo;
import ca.ubc.cs.beta.aeatk.misc.version.OSVersionInfo;
import ca.ubc.cs.beta.aeatk.misc.version.VersionTracker;
import ca.ubc.cs.beta.aeatk.misc.watch.StopWatch;
import ca.ubc.cs.beta.aeatk.options.AbstractOptions;
import ca.ubc.cs.beta.aeatk.parameterconfigurationspace.ParameterConfiguration;
import ca.ubc.cs.beta.aeatk.probleminstance.InstanceListWithSeeds;
import ca.ubc.cs.beta.aeatk.probleminstance.ProblemInstance;
import ca.ubc.cs.beta.aeatk.probleminstance.ProblemInstanceOptions.TrainTestInstances;
import ca.ubc.cs.beta.aeatk.probleminstance.ProblemInstanceSeedPair;
import ca.ubc.cs.beta.aeatk.probleminstance.seedgenerator.InstanceSeedGenerator;
import ca.ubc.cs.beta.aeatk.random.SeedableRandomPool;
import ca.ubc.cs.beta.aeatk.runhistory.RunHistory;
import ca.ubc.cs.beta.aeatk.state.StateFactoryOptions;
import ca.ubc.cs.beta.aeatk.targetalgorithmevaluator.TargetAlgorithmEvaluator;
import ca.ubc.cs.beta.aeatk.targetalgorithmevaluator.base.cli.CommandLineTargetAlgorithmEvaluatorFactory;
import ca.ubc.cs.beta.aeatk.targetalgorithmevaluator.base.cli.CommandLineTargetAlgorithmEvaluatorOptions;
import ca.ubc.cs.beta.aeatk.targetalgorithmevaluator.exceptions.TargetAlgorithmAbortException;
import ca.ubc.cs.beta.aeatk.targetalgorithmevaluator.init.TargetAlgorithmEvaluatorBuilder;
import ca.ubc.cs.beta.aeatk.termination.TerminationCondition;
import ca.ubc.cs.beta.aeatk.trajectoryfile.TrajectoryFile;
import ca.ubc.cs.beta.aeatk.trajectoryfile.TrajectoryFileEntry;

import ca.ubc.cs.beta.paramils.builder.ParamILSBuilder;
import ca.ubc.cs.beta.paramils.configurator.AbstractAlgorithmFramework;
import ca.ubc.cs.beta.paramils.misc.ParamILSOptions;
import ca.ubc.cs.beta.paramils.misc.version.ParamILSVersionInfo;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.ParameterException;

public class ParamILSExecutor {
  private static Logger log;
  private static Marker exception;
  private static Marker stackTrace;

  private static String logLocation = "<NO LOG LOCATION SPECIFIED, FAILURE MUST HAVE OCCURED EARLY>";

  private static InstanceListWithSeeds trainingILWS;
  private static InstanceListWithSeeds testingILWS;

  private static Map<String, AbstractOptions> taeOptions;
  private static SeedableRandomPool pool;

  private static String outputDir;

  /**
   * Executes ParamILS then exits the JVM {@see System.exit()}
   *
   * @param args string arguments
   */
  public static void main(String[] args) {
    int returnValue = oldMain(args);

    if (log != null) {
      log.debug("Returning with value: {}", returnValue);
    }

    System.exit(returnValue);
  }


  /**
   * Executes ParamILS according to the given arguments
   * @param args  string input arguments
   * @return return value for operating system
   */
  public static int oldMain(String[] args) {
    /*
     * WARNING: DO NOT LOG ANYTHING UNTIL AFTER WE HAVE PARSED THE CLI
     * OPTIONS AS THE CLI OPTIONS USE A TRICK TO ALLOW LOGGING TO BE
     * CONFIGURABLE ON THE CLI.
     * IF YOU LOG PRIOR TO IT ACTIVATING, IT WILL BE IGNORED
     */
    try {
      ParamILSOptions options = parseCLIOptions(args);
      AlgorithmExecutionConfiguration execConfig = options.getAlgorithmExecutionConfig();
      ParamILSBuilder paramilsBuilder = new ParamILSBuilder();
      AbstractAlgorithmFramework paramils = paramilsBuilder.getAutomaticConfigurator(execConfig, trainingILWS, options, taeOptions, outputDir, pool);
      log.info("This version of ParamILS has been revised by Gilles Ottervanger on 17-05-2021");
      log.info("The first {} parameters in the pcs file are treated as trade-off parameters.", options.nTradeoffParams);

      try {
        paramils.run();
      } finally {
        paramils.clean();
      }

      pool.logUsage();

      log.info("========================================================================");
      log.info("ParamILS has finished. Reason: {}", paramils.getTerminationReason());
      paramils.logFinalReport();
      log.info("========================================================================");

      options.doValidation = (options.validationRuns > 0) ? options.doValidation : false;
      if (options.doValidation) {
        log.info("Now starting offline validation.");

        //Don't use the same TargetAlgorithmEvaluator as above as it may have runhashcode and other crap that is probably not applicable for validation
        options.scenarioConfig.algoExecOptions.taeOpts.turnOffCrashes();

        StopWatch watch = new StopWatch();
        TargetAlgorithmEvaluator validatingTae = TargetAlgorithmEvaluatorBuilder.getTargetAlgorithmEvaluator(options.scenarioConfig.algoExecOptions.taeOpts, false, taeOptions);
        List<ProblemInstance> testInstances;
        InstanceSeedGenerator testInstanceSeedGen;
        watch.start();
        try {
          if (options.validateOnTrainingSet) {
            testInstances = trainingILWS.getInstances();
            testInstanceSeedGen = trainingILWS.getSeedGen();
            testInstanceSeedGen.reinit();
          } else {
            testInstances = testingILWS.getInstances();
            testInstanceSeedGen = testingILWS.getSeedGen();
          }
          paramils.validate(validatingTae, testInstances, testInstanceSeedGen);
        } finally {
          watch.stop();
          validatingTae.notifyShutdown();
        }

        final DecimalFormat df0 = new DecimalFormat("0");
        log.info("========================================================================");
        log.info("Validation has finished. Time used: {} s.", df0.format(watch.time()/1000.0));
        paramils.logValidationMessage();
        log.info("========================================================================");
     }

      log.info("Additional information about run {} in: {}", options.seedOptions.numRun, outputDir);

      paramilsBuilder.getEventManager().shutdown();

      return AEATKReturnValues.SUCCESS;
    } catch(Throwable t) {
      System.out.flush();
      System.err.flush();

      System.err.println("Error occurred while running ParamILS\n>Error Message:"+  t.getMessage() +  "\n>Encountered Exception:" + t.getClass().getSimpleName() +"\n>Error Log Location: " + logLocation);
      System.err.flush();

      if (log != null) {
        log.error(exception, "Message: {}",t.getMessage());

        if (t instanceof ParameterException) {
          log.info("Note that some options are read from files in ~/.aeatk/");
          log.debug("Exception stack trace", t);
        } else if (t instanceof TargetAlgorithmAbortException) {
          log.error(CommonMarkers.SKIP_CONSOLE_PRINTING, "A serious problem occured during target algorithm execution and we are aborting execution ",t );
          log.error("We tried to call the target algorithm wrapper, but this call failed.");
          log.error("The problem is (most likely) somewhere in the wrapper or with the arguments to ParamILS.");
          log.error("The easiest way to debug this problem is to manually execute the call we tried and see why it did not return the correct result");
          log.error("The required output of the wrapper is something like \"Result for ParamILS: x,x,x,x,x\".);");
          //log.error("Specifically the regex we are matching is {}", CommandLineAlgorithmRun.AUTOMATIC_CONFIGURATOR_RESULT_REGEX);
        } else {
          log.info("Maybe try running in DEBUG mode if you are missing information");
          log.error(exception, "Exception:{}", t.getClass().getCanonicalName());
          StringWriter sWriter = new StringWriter();
          PrintWriter writer = new PrintWriter(sWriter);
          t.printStackTrace(writer);
          log.error(stackTrace, "StackTrace:{}", sWriter.toString());
        }

        log.info("Exiting ParamILS with failure. Log: " + logLocation);
        log.info("For a list of available commands use:  --help");

        t = t.getCause();
      } else {
        if (t instanceof ParameterException ) {
          System.err.println(t.getMessage());
          t.printStackTrace();
        } else {
          t.printStackTrace();
        }
      }

      if (t instanceof ParameterException) {
        return AEATKReturnValues.PARAMETER_EXCEPTION;
      }

      if (t instanceof StateSerializationException) {
        return AEATKReturnValues.SERIALIZATION_EXCEPTION;
      }

      if (t instanceof TrajectoryDivergenceException) {
        return AEATKReturnValues.TRAJECTORY_DIVERGENCE;
      }

      return AEATKReturnValues.OTHER_EXCEPTION;
    }
  }


  private static String runGroupName = "DEFAULT";

  /**
   * Parsers Command Line Arguments and returns a options object
   * @param args
   * @return
   */
  private static ParamILSOptions parseCLIOptions(String[] args) throws ParameterException, IOException {
    //DO NOT LOG UNTIL AFTER WE PARSE CONFIG OBJECT

    ParamILSOptions options = new ParamILSOptions();
    taeOptions = options.scenarioConfig.algoExecOptions.taeOpts.getAvailableTargetAlgorithmEvaluators();
    JCommander jcom = JCommanderHelper.getJCommanderAndCheckForHelp(args, options, taeOptions);

    jcom.setProgramName("paramils");

    try {
      try {
        try {
          args = processScenarioStateRestore(args);
          jcom.parse(args);
        } finally {
          runGroupName = options.runGroupOptions.getFailbackRunGroup();
        }

        runGroupName = options.getRunGroupName(taeOptions.values());

        /*
         * Build the Serializer object used in the model
         */
        outputDir = options.getOutputDirectory(runGroupName);

        File outputDirFile = new File(outputDir);

        if (!outputDirFile.exists()) {
          outputDirFile.mkdirs();
          //Check again to ensure there isn't a race condition
          if (!outputDirFile.exists()) {
            throw new ParameterException("Could not create all folders necessary for output directory: " + outputDir);
          }
        }

      } finally {
        options.logOptions.initializeLogging(outputDir, options.seedOptions.numRun);
        ParamILSExecutor.logLocation = options.logOptions.getLogLocation(outputDir,options.seedOptions.numRun);

        log = LoggerFactory.getLogger(ParamILSExecutor.class);

        exception = MarkerFactory.getMarker("EXCEPTION");
        stackTrace = MarkerFactory.getMarker("STACKTRACE");

        VersionTracker.setClassLoader(SPIClassLoaderHelper.getClassLoader());

        VersionTracker.logVersions();
        ParamILSVersionInfo s = new ParamILSVersionInfo();
        JavaVersionInfo j = new JavaVersionInfo();
        OSVersionInfo o = new OSVersionInfo();
        log.info(CommonMarkers.SKIP_FILE_PRINTING,"Version of {} is {}, running on {} and {} ", s.getProductName(), s.getVersion(), j.getVersion(), o.getVersion());

        for (String name : jcom.getParameterFilesToRead()) {
          log.debug("Parsing (default) options from file: {} ", name);
        }
      }

      JCommanderHelper.logCallString(args, "paramils");

      Map<String, String> env = new TreeMap<String, String>(System.getenv());

      StringBuilder sb = new StringBuilder();
      for (String envName : env.keySet()) {
        sb.append(envName).append("=").append(env.get(envName)).append("\n");
      }

      log.info(CommonMarkers.SKIP_CONSOLE_PRINTING, "********** The next bit of output can be ignored, it is merely useful for debugging **********");
      log.info(CommonMarkers.SKIP_CONSOLE_PRINTING,"==========Environment Variables===========\n{}", sb.toString());

      Map<Object,Object > props = new TreeMap<Object, Object>(System.getProperties());
      sb = new StringBuilder();
      for (Entry<Object, Object> ent : props.entrySet()) {

        sb.append(ent.getKey().toString()).append("=").append(ent.getValue().toString()).append("\n");
      }

      String hostname = "[UNABLE TO DETERMINE HOSTNAME]";
      try {
        hostname = InetAddress.getLocalHost().getHostName();
      } catch(UnknownHostException e) {
        //If this fails it's okay we just use it to output to the log
      }

      log.info(CommonMarkers.SKIP_CONSOLE_PRINTING,"Hostname:{}", hostname);
      log.info(CommonMarkers.SKIP_CONSOLE_PRINTING,"==========System Properties==============\n{}", sb.toString() );

      JCommanderHelper.logConfigurationInfoToFile(jcom);
      pool = options.seedOptions.getSeedableRandomPool();

      TrainTestInstances tti = options.getTrainingAndTestProblemInstances(pool, new SeedableRandomPool(options.validationSeed + options.seedOptions.seedOffset,pool.getInitialSeeds()));
      trainingILWS = tti.getTrainingInstances();
      testingILWS = tti.getTestInstances();

      try {
        //We don't handle this more gracefully because this seems like a super rare incident.
        if (ManagementFactory.getThreadMXBean().isThreadCpuTimeEnabled()) {
          log.trace("JVM Supports CPU Timing Measurements");
        } else {
          log.warn("This Java Virtual Machine has CPU Time Measurements disabled, tunerTimeout will not contain any ParamILS Execution Time.");
        }
      } catch(UnsupportedOperationException e) {
        log.warn("This Java Virtual Machine does not support CPU Time Measurements, tunerTimeout will not contain any ParamILS Execution Time Information (http://docs.oracle.com/javase/1.5.0/docs/api/java/lang/management/ThreadMXBean.html#setThreadCpuTimeEnabled(boolean))");
      }

      if (options.seedOptions.numRun + options.seedOptions.seedOffset < 0) {
        log.warn("NumRun {} plus Seed Offset {} should be positive, things may not seed correctly",options.seedOptions.numRun, options.seedOptions.seedOffset );
      }

      return options;
    } catch(IOException e) {
      throw e;
    } catch(ParameterException e) {
      throw e;
    }
  }

  private static String[] processScenarioStateRestore(String[] args) {
    return StateFactoryOptions.processScenarioStateRestore(args);
  }
}
