# Allow over-ride
analysis="omics_tools"
echo "activate "${analysis}
source activate ${analysis}

if [ -z "${CONTAINER_IMAGE}" ]
then
    version=$(cat ./_util/VERSION)
    CONTAINER_IMAGE="index.docker.io/sd2e/precomputed-data-table-omics-tools:$version"
fi
. _util/container_exec.sh

log(){
    mesg "INFO" $@
}

die() {
    mesg "ERROR" $@
    ${AGAVE_JOB_CALLBACK_FAILURE}
    exit 0
}

mesg() {
    lvl=$1
    shift
    message=$@
    echo "[$lvl] $(utc_date) - $message"
}

utc_date() {
    echo $(date -u +"%Y-%m-%dT%H:%M:%SZ")
}

#### BEGIN SCRIPT LOGIC

echo "launching "${analysis}

input_counts_file=$(basename "${input_data}")

# expensive processing and/or analysis. Fail if not found.
if [ ! -f "${input_counts_file}" ];
then
    die "input data ${input_counts_file} not found or was inaccessible"
fi
container_exec ${CONTAINER_IMAGE} /opt/conda/envs/${analysis}/bin/python3 /src/${analysis}/run_omics.py --input_counts_file ${input_counts_file} --config_file ${config_file} --output_dir .