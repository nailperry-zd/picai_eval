import os
import pandas as pd

from picai_eval.metrics import *

def parse_lesion_list(metrics_json_path):
    metrics = Metrics(metrics_json_path)
    lesion_results = metrics.lesion_results
    output_file_path = os.path.splitext(metrics_json_path)[0] + '.xlsx'
    # Initialize a list to store the rows
    rows = []

    # Iterate over the lesions and classify
    for patient_id in lesion_results.keys():
        for lesion in lesion_results[patient_id]:
            # Determine the group based on volume
            if lesion[KEY_LABEL] == 1:
                group = "hit" if lesion[KEY_OVERLAP] > 0. else "miss"
            else:
                group = "overprediction"

            # Create a row
            row = {
                "patient_id": patient_id,
                "label": lesion[KEY_LABEL],
                "group": group,
                "confidence": lesion[KEY_CONFIDENCE],
                "overlap(DSC)": lesion[KEY_OVERLAP],
                "volume(cc)": lesion[KEY_VOLUME]
            }
            rows.append(row)

    # Create a DataFrame
    df = pd.DataFrame(rows)

    # Export to Excel
    df.to_excel(output_file_path, index=False)

if __name__ == '__main__':
    metrics_json_path = r"Y:\rstrial\input\images\batch1_noncropped_highb0002_registered_manual\Dataset302_rstrial_batch1\prediction0_Dataset713_picai_baseline_nnUNetTrainer_new\metrics_DSC_0.1_10cases_full_281620.json"
    parse_lesion_list(metrics_json_path)
