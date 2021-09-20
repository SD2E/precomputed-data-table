# SD2 Precomputed Data Table (PDT)

This project's goal is to improve analytical reproducibility by enabling automated and consistent execution of stereotyped analyses of different degrees of algorithmic complexity, as well as providing versioned, organized, analysis-ready, precomputed data for custom analyses.

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
```
```
abaco run -m '{"experiment_ref": "YeastSTATES-OR-Gate-CRISPR-Dose-Response", "data_converge_dir": "agave://data-sd2e-projects.sd2e-project-43/reactor_outputs/complete/YeastSTATES-OR-Gate-CRISPR-Dose-Response/20210420212751", "datetime_stamp": "20210420222801", "analysis": "fcs_signal_prediction"}' precomputed-data-table-reactor.prod
```
```
abaco run -m '{"experiment_ref": "YeastSTATES-OR-Gate-CRISPR-Dose-Response", "control_set_dir": "agave://data-sd2e-projects.sd2e-project-14/xplan-reactor/data/transcriptic", "data_converge_dir": "agave://data-sd2e-projects.sd2e-project-43/reactor_outputs/complete/YeastSTATES-OR-Gate-CRISPR-Dose-Response/20210420212751", "datetime_stamp": "20210420222801", "analysis": "live-dead-prediction"}' precomputed-data-table-reactor.prod
```
```
abaco run -m '{"experiment_ref": "YeastSTATES-OR-Gate-CRISPR-Dose-Response", "data_converge_dir": "agave://data-sd2e-projects.sd2e-project-43/reactor_outputs/complete/YeastSTATES-OR-Gate-CRISPR-Dose-Response/20210420212751", "datetime_stamp": "20210420222801", "analysis": "perform-metrics", "mtype": "FLOW"}' precomputed-data-table-reactor.prod
```
```
abaco run -m '{"experiment_ref": "YeastSTATES-OR-Gate-CRISPR-Dose-Response", "pm_batch_path": "agave://data-sd2e-projects.sd2e-project-48/complete/YeastSTATES-OR-Gate-CRISPR-Dose-Response/20210420222801", "datetime_stamp": "20210420222801", "analysis": "diagnose", "mtype": "FLOW"}' precomputed-data-table-reactor.prod
```

To manually launch Omics Tools at scale, see https://gitlab.sd2e.org/sd2program/omics_tools

### PDT Output
Output is in the precomputed_data_table project folder:

`/home/jupyter/sd2e-projects/sd2e-project-48`

each run is marked with its experiment reference and a datetime stamp 
* complete/  
    all experiments in an experiment reference passed upstream metadata check, ETL processing and validation
* preview/  
    not all experiments in an experiment reference HAVE passed upstream metadata check, ETL processing and validation
  
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
Most analyses output data frames that are written to csv or json. When possible, PDT merges or appends the output of analyses with the input. Therefore, most output files contain columns taken from the input file and these columns respective descriptions will NOT be found here. The subsequent descriptions of files or folder contents will only pertain to their respective analysis. Some of the analyses have their own gitlab repository containing the relevant information, and so their gitlab links are provided.

**Performance Metrics**

See https://gitlab.sd2e.org/gda/perform_metrics

**Diagnose**

See https://gitlab.sd2e.org/gda/diagnose

**Omics Tools**

See https://gitlab.sd2e.org/sd2program/omics_tools

**Growth Analysis**

See https://gitlab.sd2e.org/rpg/xplan-od-growth-analysis

**Wasserstein Distance Metric**
* Data types used:
  * Flow Cytometry:
    * Log10 normalized (fc_raw_log10)
    * TASBE processed and normalized (fc_etl)

* Three types of comparisons (configurations) are currently used to analyze data on a per strain or replicate basis:
  * inducer_diff: The min. inducer concentration vs max. inducer concentration for each time point sampled for each strain
  * time_reps_diff: The min. time point vs max. time point for each inducer concentration for a replicate of a strain
  * time_diff: The min. time point vs max. time point for each inducer concentration for each strain

* Files:
    * pdt_\<experiment reference>\__\<data type>\_stats_inducer_diff_params\_\<datetimestamp>.json 
    * pdt_\<experiment reference>\__\<data type>\_stats_inducer_diff_summary\_\<datetimestamp>.csv
    * pdt_\<experiment reference>\__\<data type>\_stats_time_diff_params\_\<datetimestamp>.json 
    * pdt_\<experiment reference>\__\<data type>\_stats_time_diff_summary\_\<datetimestamp>.csv
    * pdt_\<experiment reference>\__\<data type>\_stats_time_reps_diff_params\_\<datetimestamp>.json 
    * pdt_\<experiment reference>\__\<data type>\_stats_time_reps_diff_summary\_\<datetimestamp>.csv
    * pdt_\<experiment reference>\__\<data type>\_stats_wasserstein_dists\_\<datetimestamp>.csv
        * This file contains the pairwise calculation of the wasserstein distance between all samples 

* Columns (for summary files):
    * group_name: The columns used to group samples by
    * wasserstein_min: The minimum wasserstein distance
    * wasserstein_median: The median wasserstein distance
    * wasserstein_max: The maximum wasserstein distance
    * sample_id-init: One sample of the two samples in the comparison 
    * sample_id-fin: The other sample of the two samples in the comparison


**Live/Dead Prediction**

See https://gitlab.sd2e.org/sd2program/precomputed-data-table/tree/master/app/precomputed-data-table-live-dead-prediction/grouped_control_prediction for instructions on installing and using the analysis individually from the PDT.

* Data types used:
  * Flow Cytometry:
    * raw flow cytometry data (data type: fc_raw)

* Along with the file listed in Files, Live/Dead Prediction contains a directory named `runs`, which contains the model information and the training, testing, and predicted data from the analysis. 

* Files:
    * pdt_\<experiment reference>\__live_dead_prediction.csv
        * Columns:
            * RF_prediction_mean: The mean of the Random Forest prediction
            * RF_prediction_std: The standard deviation of the Random Forest prediction

**FC Signal Prediction**

* Data types used:
  * Flow Cytometry:
    * raw flow cytometry data (data type: fc_raw)

* FC Signal Prediction makes predictions on if an event is high or low, based on the high/positive and low/negative controls in the experiment. With these event-level predictions, a sample-level prediction is made and reported. As some experiments can have more than one type of positive or negative control, each combination of high/positive and low/negative controls is used in the analysis. This is indicated in the file name as `HLn` with `n` being an integer. Two models are made (and hence two preditions are made per plate and per high/low combo): 1) (fullModel) all data for the high/positive and low/negative controls are used to train a model for subsequent predictions and 2) (cleanModel) the data for the high/positive and low/negative controls are thresholded on a probability of being high or low, repsectively, while maintaining at least 10,000 events for each control type (control samples are concatenated based on control type and then the concatenated data is thresholded on. This means data is dropped agnostic to the sample id).

* For each model type (fullModel and cleanModel), two files are generated: 1) an fc_raw_log10_stats.csv file (similar to Data Converge's) is generated for each plate where each sample's predicted ON and OFF populations are segregated into two histograms and 2) an fc_meta.csv file (similar to Data Converge's) containing additional columns related to the FCS Signal Prediction analysis. One additional fc_raw_log10_stats.cs file is generated containing the word 'cleanControls'. This file contains the fcs data histograms for the concatenated controls so only one sample_id is used for each control type and this sample_id is chosen randomly. 

* Files:
    * pdt_\<experiment reference>\__<plate id>_<high/low control combination int>_fullModel_fcs_signal_prediction__fc_meta.csv
    * pdt_\<experiment reference>\__<plate id>_<high/low control combination int>_fullModel_fcs_signal_prediction__fc_raw_log10_stats.csv
    * pdt_\<experiment reference>\__<plate id>_<high/low control combination int>_cleanModel_fcs_signal_prediction__fc_meta.csv
    * pdt_\<experiment reference>\__<plate id>_<high/low control combination int>_cleanModel_fcs_signal_prediction__fc_raw_log10_stats.csv
    * pdt_\<experiment reference>\__<plate id>_<high/low control combination int>_cleanControls_fcs_signal_prediction__fc_raw_log10_stats.csv

* Columns:
    * fc_raw_log10_stats.csv files:
        * class_label: The predicted output for all events in the histogram
        * mean_log10: The mean of the log10 distribution 
        * std_log10: The standard deviation of the log10 distribution
    * fc_meta.csv files:
        * predicted_output_mean: The mean of the event-level predictions
        * predicted_output_std: The standard deviation of the event-level predictions
        * pOFF: The number of events predicted as off
        * pON: The number of events predicted as on
        * high_control: The high/positive control used
        * low_control: The low/negative control used

