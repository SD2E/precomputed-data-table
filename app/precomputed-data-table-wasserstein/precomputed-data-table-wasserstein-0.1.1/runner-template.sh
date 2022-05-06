# Allow over-ride
echo "activate "${analysis}
source activate ${analysis}

if [ -z "${CONTAINER_IMAGE}" ]
then
    version=$(cat ./_util/VERSION)
    CONTAINER_IMAGE="index.docker.io/sd2e/precomputed-data-table-wasserstein:$version"
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
inputTarball=$(basename "${inputData}")

# Double check existence of inputTarball before undertaking
# expensive processing and/or analysis. Fail if not found.
if [ ! -f "${inputTarball}" ];
then
    die "inputData ${inputTarball} not found or was inaccessible"
fi

echo "inputTarball: " ${inputTarball}
mkdir data_converge_dir
tar zxvf ${inputTarball} --directory data_converge_dir
echo "data_converge_dir" >> .agave.archive

# No need to archive the tarball and its content since they exist elsewhere
echo "${inputTarball}" >> .agave.archive
dirNames=`tar -tzf ${inputTarball} | cut -f1 -d"/" | sort -u`
for d in ${dirNames}
do
	if [ "${d}" != "record.json" ]
    then
		echo "${d}" >> .agave.archive
	fi
done

rm ${inputTarball}
echo "launching "${analysis}
echo "invoking container_exec" ${CONTAINER_IMAGE} ${experiment_ref}
container_exec ${CONTAINER_IMAGE} /opt/conda/envs/${analysis}/bin/python3 /src/run_wasserstein_tenfold_comparisons.py --experiment_ref ${experiment_ref} --data_converge_dir data_converge_dir --analysis ${analysis}
