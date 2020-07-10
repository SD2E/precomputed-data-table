# Allow over-ride
echo "activate "${analysis}
source activate ${analysis}

if [ -z "${CONTAINER_IMAGE}" ]
then
    version=$(cat ./_util/VERSION)
    CONTAINER_IMAGE="index.docker.io/sd2e/precomputed-data-table-app:$version"
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
echo "invoking container_exec" ${CONTAINER_IMAGE} ${experiment_ref}
container_exec ${CONTAINER_IMAGE} /opt/conda/envs/${analysis}/bin/python3 /src/run_pdt.py --experiment_ref ${experiment_ref} --data_converge_dir ${data_converge_dir} --analysis ${analysis} --result_parent_dir ${result_parent_dir}
