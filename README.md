# SD2 Precomputed Data Table (PDT)

This project's goal is to enable easier linking of analytical computational
tools to data in order to support automation of analyses. Along with automating analysis tools, PDT also tracks versions and provenance of the data produced from each analysis.

### Analysis tools currently in PDT and their brief descriptions:
| Tool | Description |
|-------|-------------|
| Wasserstein Distance Metric|Configuration-based comparisons of flow cytometry data distributions |
|Performance Metrics | Allows users to configure different types of questions about performance, for example fold change between ON and OFF states. Runs a suite of metrics at several thresholds and across different groupings of samples|
| Diagnose|Identifies variables associated with variation in performance, and which high or low values should be investigated. Identifies dependent variables that cause redundancies in analysis or require a change in experimental design to investigate |
|Live/Dead Prediction | Predicts live/dead cellular events to gate dead/debris cells from downstream analysis|
|Omics Tools |Configuration based differential expression and enrichment analysis of transcriptional counts data |
| FC Signal Prediction|Fluorescent output prediction at the flow cytometer event-level for samples based on negative and positive control samples|
|Growth Analysis |Estimates growth rates and doubling times of cultures and if the culture is growing or not|


## User Guide
### Install
```buildoutcfg

```
### Input Data

### Run

### PDT Output
Output is in the precomputed_data_table project folder:

`/home/jupyter/sd2e-projects/sd2e-project-48`

each run is marked with its experiment reference and a datetime stamp 
<!--- Unsure if these descriptions are accurate -->
* testing/  
    <!--- this seems to only contain a test.txt file -->
* complete/  
    all experiments in an experiment reference passed upstream validation
* preview/  
    experiments in an experiment reference HAVE NOT passed upstream validation
  
Within each datetime stamp folder for an experiment reference, a folder will be created for each analysis that was run on the respective experiment reference. 

Also included with the analysis folders is a file named `record.json`. This file contains:
* 'precomputed_data_table version' : The version of PDT at the time the experiment reference analyzed
* 'experiment_reference' : The experiment reference analyzed
* 'date_run' : The date and time on which the experiment reference was analyzed
* 'output_dir' : The location of the PDT output for the experiment reference
* 'analyses' : This contains information on what analyses were run on the experiment reference, the input and output files for each analysis, and hashes for each output file.
* 'status_upstream'
    * 'data-converge directory' : The location of the experiment reference's data used by the PDT

### Analysis Output
Most analyses output data frames that are written to csv or json. When possible, PDT merges or appends the output of analyses with the input. Therefore, most output files contain columns taken from the input file and these columns respective descriptions will NOT be found here. The subsequent descriptions of files or folder contents will only pertain to their respective analysis.


**Wasserstein Distance Metric**
* Data Types:
  * Flow Cytometry:
    * Log10 normalized
    * ETL normalized <!--- Could be more specific -->
* Three types of comparisons (configurations) are currently used to analyze data on a per strain or replicate basis:
  * The min. inducer concentration vs max. inducer concentration for each time point sampled for each strain
  * The min. time point vs max. time point for each inducer concentration for a replicate of a strain
  * 
* Files:
    * pdt_\<experiment reference>\__\<data type>\_stats_inducer_diff_params\_\<datetimestamp>.json 
    * pdt_\<experiment reference>\__\<data type>\_stats_inducer_diff_summary\_\<datetimestamp>.csv
    * pdt_\<experiment reference>\__\<data type>\_stats_time_diff_params\_\<datetimestamp>.json 
    * pdt_\<experiment reference>\__\<data type>\_stats_time_diff_summary\_\<datetimestamp>.csv
    * pdt_\<experiment reference>\__\<data type>\_stats_reps_diff_params\_\<datetimestamp>.json 
    * pdt_\<experiment reference>\__\<data type>\_stats_reps_diff_summary\_\<datetimestamp>.csv
    * pdt_\<experiment reference>\__\<data type>\_stats_wasserstein_dists\_\<datetimestamp>.csv

**Performance Metrics**

**Diagnose**

**Live/Dead Prediction**

**Omics Tools**

**FC Signal Prediction**

**Growth Analysis**
