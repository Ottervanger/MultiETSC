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
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ca.ubc.cs.beta.aeatk.parameterconfigurationspace.ParameterConfiguration;
import ca.ubc.cs.beta.aeatk.parameterconfigurationspace.ParameterConfiguration.ParameterStringFormat;

public class ConfigWriter {
  private final FileWriter fileWriter;
  private final String fileNamePrefix;
  private static final Logger log = LoggerFactory.getLogger(ConfigWriter.class);

  public ConfigWriter(String fileNamePrefix) {
    this.fileNamePrefix = fileNamePrefix;
    try {
      fileWriter = new FileWriter(fileNamePrefix + ".txt");
      File f = new File(fileNamePrefix);
    } catch (IOException e) {
      throw new IllegalStateException("Error occured creating files",e);
    }
  }

  /**
   * Writes the configuration to the file
   *
   * @param configuration  configuration
   */
  public void writeConfig(ParameterConfiguration config) {
    String paramString = config.getFormattedParameterString(ParameterStringFormat.NODB_SYNTAX);
    log.trace("Logging configuration: {}", paramString.trim());
    try {
      fileWriter.write(paramString + "\n");
      fileWriter.flush();
    } catch(IOException e) {
      throw new IllegalStateException("Could not write final file", e);
    }
  }
}
