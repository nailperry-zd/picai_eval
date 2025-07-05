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

    # ======= NEW FEATURES ========
    num_label_1 = (df["label"] == 1).sum()
    num_hit = (df["group"] == "hit").sum()
    median_overlap_hit = df.loc[df["group"] == "hit", "overlap(DSC)"].median()

    print(f"Number of rows with label == 1: {num_label_1}")
    print(f"Number of rows where group == 'hit': {num_hit}")
    print(f"Median overlap(DSC) for 'hit' group: {median_overlap_hit:.4f}")
    # =============================

    # Export to Excel
    df.to_excel(output_file_path, index=False)

if __name__ == '__main__':
    metrics_json_path = r"Y:\picai\workdir\nnUNet_results\nnUNet\3d_fullres\Task822_PICAI_Expert\nnUNetTrainerV2_Loss_FL_and_CE_checkpoints_FL0__nnUNetPlansv2.1\fold_0\validation_raw-best\metrics_DSC_0.1_44cases_full_67335.json"
    parse_lesion_list(metrics_json_path)
