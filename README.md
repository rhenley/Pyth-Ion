# Pyth-Ion

Pyth-ion is a python based program designed for analysis of nanopore ionic current data. 

## Capabilities

The software provides a viewer for a variety of file types including .log (Chimera Amplifiers), .abf (Axopatch), .opt (General Binary), or .csv (standard text files). Assumptions are made in regards to the structure of each of these as the program requires information on sampling rate and low-pass filter value, so some type of supplementary file containing settings is usually assumed (the .log file usually has an accompanying .mat for example, for .abf these are contained in the header).

## Installation

This package can be easily run from a standard Anaconda 3 distribution, the only additional requirement being pyqtgraph
