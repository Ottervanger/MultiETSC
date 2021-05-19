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

package ca.ubc.cs.beta.paramils.exceptions;

public class TargetAlgorithmExecutionException extends ParamILSRuntimeException {

  private static final long serialVersionUID = -8701769256842561265L;

  public TargetAlgorithmExecutionException(String message) {
    super(message);
  }
  public TargetAlgorithmExecutionException(Throwable t) {
    super(t);
  }

}
