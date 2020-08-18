Introduction
==========

The `pysd2cat` circuit analysis tool analyzes flow cytometry data to perform the following tasks:

* Predict Live and Dead cells
* Compute the accuracy of a circuit wrt. a GFP threshold


Quick Start:
=========
Currently, `pysd2cat` runs on the TACC infrastructure so that it may have fast access to data. It assumes a root directory (containing data files) exists on the host at: `/work/projects/SD2E-Community/prod/data/uploads/`.

* Clone the repo: `https://gitlab.sd2e.org/dbryce/pysd2cat.git`
* `cd pysd2cat`
* `python setup.py install`

To run an example analysis script, run:
* `python src/pysd2cat/analysis/live_dead_analysis.py`

Environment Configuration:
=====================
You will need to have the following dependencies and logins:
1. pip install transcriptic
2. pip install autoprotocol
3. Get a transcriptic login from Transcriptic (or Strateos). Talk to Josh Nowak.
4. Run transcriptic login once so that it creates the required folders to fetch data.

Code Layout:
===========

The source code is divided into subdirectories, as follows:

* `src/pysd2cat/data`: routines to acquire data and metadata
* `src/pysd2cat/analysis`: routines to analyze data
* `src/pysd2cat/plot`: routines to plot data