# Allow over-ride
echo "activate "${analysis}
source activate ${analysis}

if [ -z "${CONTAINER_IMAGE}" ]
then
    version=$(cat ./_util/VERSION)
<<<<<<< HEAD
    CONTAINER_IMAGE="index.docker.io/sd2e/precomputed-data-table-growth-analysis:$version"
=======
<<<<<<< HEAD:app/precomputed-data-table-wasserstein/precomputed-data-table-wasserstein-0.1.0/runner-template.sh
    CONTAINER_IMAGE="index.docker.io/sd2e/precomputed-data-table-wasserstein:$version"
=======
    CONTAINER_IMAGE="index.docker.io/sd2e/precomputed-data-table-growth-analysis:$version"
>>>>>>> f3cc67d07ca172c0c5754f1cb3cbe270077b61d3:app/precomputed-data-table-growth-analysis/precomputed-data-table-growth-analysis-0.1.0/runner-template.sh
>>>>>>> f3cc67d07ca172c0c5754f1cb3cbe270077b61d3
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
echo "invoking container_exec" ${CONTAINER_IMAGE} ${experiment_ref}
<<<<<<< HEAD
container_exec ${CONTAINER_IMAGE} /opt/conda/envs/${analysis}/bin/python3 /src/run_pdt.py --experiment_ref ${experiment_ref} --data_converge_dir ${data_converge_dir} --analysis ${analysis}
=======
container_exec ${CONTAINER_IMAGE} /opt/conda/envs/${analysis}/bin/python3 /src/run_od_growth_analysis.py --experiment_ref ${experiment_ref} --data_converge_dir ${data_converge_dir} --analysis ${analysis}
>>>>>>> f3cc67d07ca172c0c5754f1cb3cbe270077b61d3
