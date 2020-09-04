class Names:
    index = "arbitrary_index"
    time = "time_point"
    stain = "stain"
    label = "label"
    label_preds = "label_predictions"
    cluster_preds = "cluster_predictions"
    data_file_name = "pipeline_data.csv"
    # data_file_name = "sampled_data_for_testing.csv"

    # strains
    yeast = "yeast"
    bacillus = "bacillus"
    ecoli = "ecoli"

    # treatments
    ethanol = "ethanol"
    heat = "heat"
    treatments_dict = {
        ethanol: {yeast: [0, 10, 15, 20, 80],
                  bacillus: [0, 5, 10, 15, 40],
                  ecoli: [0, 5, 10, 15, 40]},
        heat: [0]
    }
    time_points = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    # feature columns
    morph_cols = ["FSC-A", "FSC-H", "FSC-W", "SSC-A", "SSC-H", "SSC-W"]
    sytox_cols = ["RL1-A", "RL1-H", "RL1-W"]
    bl_cols = ["BL1-A", "BL1-H", "BL1-W"]
    # mito_cols = None
    morph_cols = ["log_{}".format(x) for x in morph_cols]
    sytox_cols = ["log_{}".format(x) for x in sytox_cols]
    bl_cols = ["log_{}".format(x) for x in bl_cols]

    # experiment dictionary
    exp_dict = {
        (yeast, ethanol): "temporary_yeast_ethanol",
        (bacillus, ethanol): "temporary_bacillus_ethanol",
        (ecoli, ethanol): "temporary_ecoli_ethanol",
        # (bacillus, ethanol): "experiment.transcriptic.r1eaf248xavu8a",
        # (ecoli, ethanol): "experiment.transcriptic.r1eaf25ne8ajts"
    }
    # each experiment should have a corresponding folder with the same name as the experiment_id
    # inside the folder you will have data files: dataset, train, test, normalized_train, normalized_test, etc.
    # then the LiveDeadPipeline can call file if it exists or otherwise create it using preprocessing methods
    exp_data_dir = "experiment_data"
    harness_output_dir = "test_harness_outputs"
    pipeline_output_dir = "pipeline_outputs"

    num_live = "num_live"
    num_dead = "num_dead"
    percent_live = "predicted %live"

    # labeling methods:
    thresholding_method = "thresholding_method"
    condition_method = "condition_method"
    cluster_method = "cluster_method"
