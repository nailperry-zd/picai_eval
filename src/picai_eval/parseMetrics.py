from picai_eval import Metrics
import openpyxl

if __name__ == '__main__':
    # metrics_json_dir = r"C:\Users\dzha937\DEV\RSTrial\Dataset713_picai_baseline\nnunetv2-vanilla-prediction0"
    # metrics_json_dir = r"C:\Users\dzha937\DEV\RSTrial\Ella_cropped"
    metrics_json_dir = r"C:\Users\dzha937\DEV\RSTrial\Dataset713_picai_baseline\nnUNetTrainerV2_Loss_FL_and_CE_checkpoints"
    # metrics_json_dir = r"C:\Users\dzha937\DEV\PICAI\Dataset713_picai_baseline\val"
    metrics_json_path = rf"{metrics_json_dir}\metrics-model_best-0-dynamic_IoU.json"
    # metrics_json_path = rf"{metrics_json_dir}\metrics_DSC_509849.json"
    # metrics_json_path = rf"{metrics_json_dir}\metrics_IoU_270247.json"
    metrics_json_excel_path = rf"{metrics_json_dir}\metrics-model_best-0-lesions_IoU_whole.xlsx"
    metrics = Metrics(metrics_json_path)
    # print(f'lesion-level metrics.AP={metrics.AP}')
    # print(f'patient-level metrics.auroc={metrics.auroc}')

    lesion_results = metrics.lesion_results
    lesion_results_keys = lesion_results.keys()
    lesion_index = 1
    lesion_list = []
    for lesion_results_key in lesion_results_keys:
        tem_list = lesion_results[lesion_results_key]
        for lesion in tem_list:
            lesion_entry = dict()
            # if lesion[0] == 1:
            lesion_entry['index'] = lesion_index
            lesion_entry['patient_id'] = lesion_results_key
            lesion_entry['overlap'] = lesion[2]
            lesion_entry['confidence'] = lesion[1]
            lesion_entry['pred'] = 1 if lesion[1] > 0.5 else 0
            lesion_entry['label'] = lesion[0]
            print(f'lesion {lesion_index}, {lesion_results_key}, overlap: {lesion[2]}, confidence: {lesion[1]}')
            lesion_list.append(lesion_entry)
            lesion_index = lesion_index + 1



    # Create a new workbook
    workbook = openpyxl.Workbook()

    # Get the active worksheet
    worksheet = workbook.active

    # Write the header row
    worksheet['A1'] = 'Index'
    worksheet['B1'] = 'Patient ID'
    worksheet['C1'] = 'Overlap'
    worksheet['D1'] = 'Confidence'
    worksheet['E1'] = 'Prediction'
    worksheet['F1'] = 'Label'

    # Write the data to the worksheet
    for row_index, entry in enumerate(lesion_list, start=2):
        for col_index, (item, value) in enumerate(entry.items(), start=1):
            worksheet.cell(row=row_index, column=col_index, value=value)

    # Save the workbook to a file
    workbook.save(metrics_json_excel_path)


