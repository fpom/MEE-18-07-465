# Using discrete systems to exhaustively characterize the dynamics of an integrated ecosystem: implementation and models

> Authors: Cédric Gaucherel, Franck Pommereau
> Journal: Methods in Ecology and Evolution

## Content

 * file `termites-ca.rr` holds the termites model discussed throughout the paper
 * file `termites-ac.pnml` is the corresponding Petri net in PNML format
   (see http://www.pnml.org/)
 * the Python implementation of the analysis method is in files `*.py` and `ktz.pyx`
 * directory `ktzlib` contains C files to read Tina `KTZ` format (courtesy of Bernard Berthomieu)
 * `Makefile` allows to build the software and produce the figures for the paper

## Dependencies

 * Python 2.7: https://www.python.org/
 * Cython: https://cython.org/
 * matplotlib: https://matplotlib.org/
 * pydot: https://github.com/pydot/pydot
 * NetworkX: https://networkx.github.io/
 * SNAKES: http://snakes.ibisc.univ-evry.fr/
 * pandas: https://pandas.pydata.org/
 * Tina: http://projects.laas.fr/tina/

## Installation

No installation is required (and no `setup.py` is provided), just run `make ktz.so` to build the Cython module `ktz` that is needed.

## Usage

Run `python econet.py` to get help about the available commands.
Then run `python econet.py MODEL.rr COMMAND...` to execute commands as instructed in the help.
 
Run `make figs` to build the figures used in the paper.
Looking into file `Makefile` may help to see how commands are used.

## Licence

File `termites.rr` is released under the terms of Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)

The source code is released under the GNU General Public Licence version 3

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
