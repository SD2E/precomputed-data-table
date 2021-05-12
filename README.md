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
| FCS Signal Prediction|Fluorescent output prediction at the flow cytometer event-level for samples based on negative and positive control samples|
|Growth Analysis |Estimates growth rates and doubling times of cultures and if the culture is growing or not|


## User Guide
### Build
There is a `Makefile` in the project folder that can be used to create the Docker images for the PDT reactor as well as each of the analysis modules listed above, except for `Performance Metrics` and `Diagnose`, which are two external applications covered in https://gitlab.sd2e.org/gda/perform_metrics.git and https://gitlab.sd2e.org/gda/diagnose.git. To build the PDT reactor:

```
cd precomputed-data-table
make reactor-image
abaco deploy -u precomputed-data-table-reactor.prod
```
where `precomputed-data-table-reactor.prod` is the alias to the production instance of the PDT reactor. You can replace it with another reactor id or simply use `abaco deploy` to create a new reactor instance.

The following commands use `FCS Signal Prediction` to show how to build the Docker image of an analysis and deploy it to TACC.

```
cd precomputed-data-table
make fcs-signal-prediction-image
make deploy-fcs-signal-prediction
```

### Input Data
| Tool | experiment_id | experiment_ref | input_dir | config_file | data_converge_dir | datetime_stamp | control_set_dir|mtypes | pm_batch_path |
|-------|------------|-------------|------|------|----------|----------|-----|---------|--------|
|Wasserstein Distance Metric|  | :heavy_check_mark: | | | :heavy_check_mark: | :heavy_check_mark: |  |  |  |  |
|Performance Metrics |  | :heavy_check_mark: | | |:heavy_check_mark: | :heavy_check_mark: |  |:heavy_check_mark: |  | |
|Diagnose| | :heavy_check_mark: | | | :heavy_check_mark: | :heavy_check_mark: |  | :heavy_check_mark:| :heavy_check_mark: |  |
|Live/Dead Prediction | | :heavy_check_mark: | | | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | | | |
|Omics Tools | :heavy_check_mark: | |:heavy_check_mark: |:heavy_check_mark: | | | | | | |
|FCS Signal Prediction| | :heavy_check_mark: | | | :heavy_check_mark: | :heavy_check_mark: | | | | |
|Growth Analysis | | :heavy_check_mark: | | | :heavy_check_mark: | :heavy_check_mark: | | | | |

#### Notes:
* `input_dir` is used by Omics Tools and should point to the output folder containing RNASeq pipeline output, matching the `archive_path` field of an entry in the `jobs` table, e.g., `/products/v2/1068bfdb0f2a53f1a97eb08c946ee732/OZ6vNeWjwAnkPGxO63ZoApyJ/PAVJ5WXoBNGrBjZvQjjdoAxe`
* `config_file` is used by Omics Tools and is the path to the configuration file in the omics_tools repo. Because these files are copied into the Docker container under `/src`, it needs to start with the leading `/`, e.g., `/src/omics_tools/config/NAND_2_0.json`
* `data_converge_dir` is needed by all analysis tools except Omics Tools. It is the Agave path to the folder containing the Data Converge output, e.g., `agave://data-sd2e-projects.sd2e-project-43/reactor_outputs/preview/Microbe-LiveDeadClassification/20200721171808`
* `control_set_dir` is used by Live/Dead Prediction and it is the Agave path to the folder containing the control FCS files, i.e., `agave://data-sd2e-projects.sd2e-project-14/xplan-reactor/data/transcriptic`
* `mtypes` is used by both Performance Metrics and Diagnose. It is a summary string indicating the measurement types contained in experiments associated with experiment_ref, e.g., `PF---`, where P is for PLATE_READER and F is for FLOW
* `pm_batch_Path` is used by Diagnose and is the Agave path to the folder containing the Performance Metrics output, e.g., `agave://data-sd2e-projects.sd2e-project-48/preview/YeastSTATES-1-0-Time-Series-Round-3-0/20201216224304`

### Run
Except for Omics Tools, which requires manual launch, all other PDT analysis tools have been integrated into the automated pipeline so they are launched automatically, as soon as Data Converge has finished processing the corresponding experiment_ref. In the case of Diagnose, it is launched as soon as Performance Metrics has finished processing the data. In some cases, it may be desirable to manually launch these tools, e.g., if a change has been made in the analysis itself to test out a modified algorithm. There are two ways to do this. The first is to use the PDT reactor to launch the analysis. The advantage of this approach is that all the input data will be automatically staged if running in the TACC infrastructure. The second approach is to bypass the reactor and launch the analysis directly. This would usually require inspecting the corresponding `runner-template.sh` script to see how the input data are staged so they can be manually staged. Once input data are staged and accessible by the analysis code, the corresponding entry python script can be launched by providing the command line input arguments listed in the above table. Below are examples showing how to launch some of these analysis tools via the PDT reactor:

```
abaco run -m '{"experiment_id":"experiment.ginkgo.29422", "input_dir": "/products/v2/1068bfdb0f2a53f1a97eb08c946ee732/OZ6vNeWjwAnkPGxO63ZoApyJ/PAVJ5WXoBNGrBjZvQjjdoAxe", "config_file": "/src/omics_tools/tests/config/Bacillus_Inducer_1_0.json", "analysis":"omics_tools"}' precomputed-data-table-reactor.prod

abaco run -m '{"experiment_ref": "YeastSTATES-OR-Gate-CRISPR-Dose-Response", "data_converge_dir": "agave://data-sd2e-projects.sd2e-project-43/reactor_outputs/complete/YeastSTATES-OR-Gate-CRISPR-Dose-Response/20210420212751", "datetime_stamp": "20210420222801", "analysis": "fcs_signal_prediction"}' precomputed-data-table-reactor.prod

abaco run -m '{"experiment_ref": "YeastSTATES-OR-Gate-CRISPR-Dose-Response", "control_set_dir": "agave://data-sd2e-projects.sd2e-project-14/xplan-reactor/data/transcriptic", "data_converge_dir": "agave://data-sd2e-projects.sd2e-project-43/reactor_outputs/complete/YeastSTATES-OR-Gate-CRISPR-Dose-Response/20210420212751", "datetime_stamp": "20210420222801", "analysis": "live-dead-prediction"}' precomputed-data-table-reactor.prod

abaco run -m '{"experiment_ref": "YeastSTATES-OR-Gate-CRISPR-Dose-Response", "data_converge_dir": "agave://data-sd2e-projects.sd2e-project-43/reactor_outputs/complete/YeastSTATES-OR-Gate-CRISPR-Dose-Response/20210420212751", "datetime_stamp": "20210420222801", "analysis": "perform-metrics", "mtype": "FLOW"}' precomputed-data-table-reactor.prod

abaco run -m '{"experiment_ref": "YeastSTATES-OR-Gate-CRISPR-Dose-Response", "pm_batch_path": "agave://data-sd2e-projects.sd2e-project-48/complete/YeastSTATES-OR-Gate-CRISPR-Dose-Response/20210420222801", "datetime_stamp": "20210420222801", "analysis": "diagnose", "mtype": "FLOW"}' precomputed-data-table-reactor.prod
```

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
