from picai_eval.analysis_utils import calculate_dsc
import SimpleITK as sitk


y_true_path = r"Y:\rstrial\input\images\batch1_noncropped_highb0002_resampled_to_first_scan_newgt_registeredinput_manual_302\Dataset302_rstrial_batch1\labelsTr\prostate_028.nii.gz"
y_pred_path = r"Y:\rstrial\output\Dataset713_picai_baseline\nnUNetTrainer\prediction0_nocropping_final_registeredinput_manual\prostate_028.nii.gz"
# Load the first NIfTI file
y_true_img = sitk.ReadImage(y_true_path)
y_true = sitk.GetArrayFromImage(y_true_img)

# Load the second NIfTI file
y_pred_img = sitk.ReadImage(y_pred_path)
y_pred = sitk.GetArrayFromImage(y_pred_img)

dsc = calculate_dsc(y_pred, y_true)
print(f'dsc is {dsc}')