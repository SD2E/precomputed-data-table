"""
gather record product info and make data files summaries for this run of precomputed data table

:authors: robert c. moseley (robert.moseley@duke.edu) and  anastasia deckard (anastasia.deckard@geomdata.com)
"""
import os
from datetime import datetime

# At runtime the version info can be retrieved from version.txt
def make_product_record(exp_ref, out_dir, dc_dir):

    datetime_stamp = datetime.now().strftime('%Y%m%d%H%M%S')
    version_file = open("/version.txt", "r")
    version_info = version_file.read().rstrip()
    print("version_info: {}".format(version_info))

    record = {
        "precomputed_data_table version": version_info,
        "experiment_reference": exp_ref,
        "date_run": datetime_stamp,
        "output_dir": out_dir,
        "analyses": {},
        "status_upstream": {
            'data-converge directory': dc_dir
        }
    }

    return record

# File version.txt can be created during make using the main function
def main():

    stream = os.popen('git show -s --format="gitinfo: %h %ci"')
    output = stream.read().strip()
    if output.startswith('gitinfo:'):
        output = output.replace('gitinfo: ', '')
    else:
        output = "NA"

    version_info = output

    print(version_info)
    
if __name__ == '__main__':
    main()