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

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Map.Entry;

import net.jcip.annotations.NotThreadSafe;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ca.ubc.cs.beta.aeatk.algorithmexecutionconfiguration.AlgorithmExecutionConfiguration;
import ca.ubc.cs.beta.aeatk.algorithmrunresult.AlgorithmRunResult;
import ca.ubc.cs.beta.aeatk.algorithmrunresult.RunStatus;
import ca.ubc.cs.beta.aeatk.exceptions.DuplicateRunException;
import ca.ubc.cs.beta.aeatk.objectives.OverallObjective;
import ca.ubc.cs.beta.aeatk.objectives.RunObjective;
import ca.ubc.cs.beta.aeatk.parameterconfigurationspace.ParameterConfiguration;
import ca.ubc.cs.beta.aeatk.probleminstance.ProblemInstance;
import ca.ubc.cs.beta.aeatk.probleminstance.ProblemInstanceSeedPair;
import ca.ubc.cs.beta.aeatk.probleminstance.seedgenerator.InstanceSeedGenerator;
import ca.ubc.cs.beta.aeatk.runhistory.KeyObjectManager;
import ca.ubc.cs.beta.aeatk.runhistory.RunData;
import ca.ubc.cs.beta.aeatk.runhistory.RunHistory;

public class ParamILSRunHistory implements RunHistory {

  private final KeyObjectManager<ParameterConfiguration> paramConfigurationList = new KeyObjectManager<ParameterConfiguration>();


  /* ======================================================================
   * Override
   * ====================================================================== */

  public void append(AlgorithmRunResult run) {
    ParameterConfiguration config = run.getAlgorithmRunConfiguration().getParameterConfiguration();
    paramConfigurationList.getOrCreateKey(config);
  }

  public int getThetaIdx(ParameterConfiguration config) {
    Integer thetaIdx = paramConfigurationList.getKey(config);
    if(thetaIdx == null)
      return -1;
    else
      return thetaIdx;
  }

  public int getOrCreateThetaIdx(ParameterConfiguration config) {
    return paramConfigurationList.getOrCreateKey(config);
  }

  /* ======================================================================
   * Sorry not implemented :/
   * ====================================================================== */

  public RunObjective getRunObjective() {
    throw new UnsupportedOperationException("Not Implemented");
  }

  public OverallObjective getOverallObjective() {
    throw new UnsupportedOperationException("Not Implemented");
  }

  public void incrementIteration() {
    throw new UnsupportedOperationException("Not Implemented");
  }

  public int getIteration() {
    throw new UnsupportedOperationException("Not Implemented");
  }

  public Set<ProblemInstance> getProblemInstancesRan(ParameterConfiguration config) {
    throw new UnsupportedOperationException("Not Implemented");
  }

  public Set<ProblemInstanceSeedPair> getProblemInstanceSeedPairsRan(ParameterConfiguration config) {
    throw new UnsupportedOperationException("Not Implemented");
  }

  public double getEmpiricalCostLowerBound(ParameterConfiguration config, Set<ProblemInstance> instanceSet, double cutoffTime) {
    throw new UnsupportedOperationException("Not Implemented");
  }

  public double getEmpiricalCostUpperBound(ParameterConfiguration config, Set<ProblemInstance> instanceSet, double cutoffTime) {
    throw new UnsupportedOperationException("Not Implemented");
  }

  public double getEmpiricalCost(ParameterConfiguration config, Set<ProblemInstance> instanceSet, double cutoffTime) {
    throw new UnsupportedOperationException("Not Implemented");
  }

  public double getEmpiricalCost(ParameterConfiguration config, Set<ProblemInstance> instanceSet, double cutoffTime,Map<ProblemInstance, Map<Long, Double>> hallucinatedValues) {
    throw new UnsupportedOperationException("Not Implemented");
  }

  public double getEmpiricalCost(ParameterConfiguration config,
                          Set<ProblemInstance> instanceSet, double cutoffTime,
                          Map<ProblemInstance, Map<Long, Double>> hallucinatedValues,
                          double minimumResponseValue) {
    throw new UnsupportedOperationException("Not Implemented");
  }

  public double getEmpiricalCost(ParameterConfiguration config,Set<ProblemInstance> instanceSet, double cutoffTime, double minimumResponseValue) {
    throw new UnsupportedOperationException("Not Implemented");
  }

  public double getTotalRunCost() {
    throw new UnsupportedOperationException("Not Implemented");
  }

  public Set<ProblemInstance> getUniqueInstancesRan() {
    throw new UnsupportedOperationException("Not Implemented");
  }

  public Set<ParameterConfiguration> getUniqueParamConfigurations() {
    throw new UnsupportedOperationException("Not Implemented");
  }

  public int[][] getParameterConfigurationInstancesRanByIndexExcludingRedundant() {
    throw new UnsupportedOperationException("Not Implemented");
  }

  public List<ParameterConfiguration> getAllParameterConfigurationsRan() {
    throw new UnsupportedOperationException("Not Implemented");
  }

  public double[][] getAllConfigurationsRanInValueArrayForm() {
    throw new UnsupportedOperationException("Not Implemented");
  }

  public List<RunData> getAlgorithmRunDataExcludingRedundant() {
    throw new UnsupportedOperationException("Not Implemented");
  }

  public List<RunData> getAlgorithmRunDataIncludingRedundant() {
    throw new UnsupportedOperationException("Not Implemented");
  }

  public List<AlgorithmRunResult> getAlgorithmRunsExcludingRedundant() {
    throw new UnsupportedOperationException("Not Implemented");
  }

  public List<AlgorithmRunResult> getAlgorithmRunsIncludingRedundant() {
    throw new UnsupportedOperationException("Not Implemented");
  }

  public int getTotalNumRunsOfConfigExcludingRedundant(ParameterConfiguration config) {
    throw new UnsupportedOperationException("Not Implemented");
  }

  public int getTotalNumRunsOfConfigIncludingRedundant(ParameterConfiguration config) {
    throw new UnsupportedOperationException("Not Implemented");
  }

  public List<AlgorithmRunResult> getAlgorithmRunsExcludingRedundant(ParameterConfiguration config) {
    throw new UnsupportedOperationException("Not Implemented");
  }

  public List<AlgorithmRunResult> getAlgorithmRunsIncludingRedundant(ParameterConfiguration config) {
    throw new UnsupportedOperationException("Not Implemented");
  }

  public Set<ProblemInstanceSeedPair> getEarlyCensoredProblemInstanceSeedPairs(ParameterConfiguration config) {
    throw new UnsupportedOperationException("Not Implemented");
  }

  public int getNumberOfUniqueProblemInstanceSeedPairsForConfiguration( ParameterConfiguration config) {
    throw new UnsupportedOperationException("Not Implemented");
  }

  public Map<ProblemInstance, LinkedHashMap<Long, Double>> getPerformanceForConfig(ParameterConfiguration configuration) {
    throw new UnsupportedOperationException("Not Implemented");
  }

  public List<Long> getSeedsUsedByInstance(ProblemInstance pi) {
    throw new UnsupportedOperationException("Not Implemented");
  }
}
