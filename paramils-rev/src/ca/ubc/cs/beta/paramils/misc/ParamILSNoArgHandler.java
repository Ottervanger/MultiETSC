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

import ca.ubc.cs.beta.aeatk.misc.options.NoArgumentHandler;

public class ParamILSNoArgHandler implements NoArgumentHandler {

  @Override
  public boolean handleNoArguments() {
    StringBuilder sb = new StringBuilder();

    //sb.append("Here goes the doc when no argument is provided\n");

    sb.append("ParamILS (http://www.cs.ubc.ca/labs/beta/Projects/ParamILS/) is an automatic configurator that allows users to automatically tune algorithm configuration spaces.");
    sb.append("\n\n");

    sb.append("  Basic Usage:\n");
    sb.append("  paramils --scenario-file <file> \n\n");

    sb.append("  Skipping Validation:\n");
    sb.append("  paramils --scenario-file <file> --validation false\n\n");

    sb.append("  Full version information is available with :\n");
    sb.append("  paramils -v\n\n");

    sb.append("  A full command line reference is available with:\n");
    sb.append("  paramils --help\n\n");

    System.out.println(sb.toString());
    return true;
  }
}
