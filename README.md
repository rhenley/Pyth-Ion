# Pyth-Ion

Pyth-ion is a python based program designed for analysis of nanopore ionic current data. 

## Capabilities
 
### Supported File Types ###
The software provides a viewer for a variety of file types including .log (Chimera Amplifiers), .abf (Axopatch), .opt (General Binary), or .csv (standard text files). Assumptions are made in regards to the structure of each of these as the program requires information on sampling rate and low-pass filter value, so some type of supplementary file containing settings is usually assumed (the .log file usually has an accompanying .mat for example, for .abf these are contained in the header).

*If any assistance is required in tailoring the software to your file type, please email roberthenley89@gmail.com, or report an issue here on Github*

### Translocation Event Detection ###
Two methods are provided for analyzing translocation data, the first being the standard event detection analysis. The software does a primitive baseline detection but for accurate analysis a user set baseline is required. For more information see the user guide. After this simple event detection, a database file is generated in the same folder as the analyzed file. This file has the same name as the analyzed file, followed by DB.txt (i.e. mydata.log generates a file named mydataDBl.log).

## Installation

This package can be easily run from a standard Anaconda 3 distribution, the only additional requirement being pyqtgraph
