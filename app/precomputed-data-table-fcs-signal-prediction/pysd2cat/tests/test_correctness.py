from pysd2cat.data import pipeline
from pysd2cat.analysis import threshold

def main():
    ex_id = 'experiment.transcriptic.r1c5va879uaex_r1c639xp952g4'
    samples = pipeline.get_experiment_samples(ex_id, 'FCS')
    sample_records = pipeline.get_metadata_dataframe(samples)
    print("sample_records.columns: {}".format(sample_records.columns))
    print(sample_records[['experiment_id','filename']].head(5))
    data_df = pipeline.get_xplan_data_and_metadata_df(sample_records, '', max_records=3000)
    print("data_df.columns: {}".format(data_df.columns))
    print(data_df.head(5))

    correctness_df = threshold.compute_correctness(data_df, high_control="NOR 00 Control", id_name='sample_id')
    print("correctness_df")
    print(correctness_df.columns.tolist())
    print(correctness_df.head(5))
    correctness_df.to_csv("samples_labelled.csv")
    
if __name__ == '__main__':
    main()