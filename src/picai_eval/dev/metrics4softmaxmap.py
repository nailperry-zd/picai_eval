import numpy as np
import SimpleITK as sitk
from picai_eval.analysis_utils import calculate_dsc
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt

def calculate_dsc_as_a_whole(y_true_path, y_pred_path):
    # Load the first NIfTI file
    y_true_img = sitk.ReadImage(y_true_path)
    y_true = sitk.GetArrayFromImage(y_true_img)

    # Load the second NIfTI file
    y_pred_img = sitk.ReadImage(y_pred_path)
    y_pred = sitk.GetArrayFromImage(y_pred_img)

    dsc = calculate_dsc(y_pred, y_true)
    print(f'dsc is {dsc:.3f}')

def cal_auroc(y_true_path, y_pred_path):
    # Load the first NIfTI file
    y_true_img = sitk.ReadImage(y_true_path)
    gt_map = sitk.GetArrayFromImage(y_true_img)

    # Load the second NIfTI file
    y_pred_img = sitk.ReadImage(y_pred_path)
    pred_map = sitk.GetArrayFromImage(y_pred_img)

    # Flatten the maps to 1D arrays
    pred_flat = pred_map.flatten()
    gt_flat = gt_map.flatten()

    # Calculate the AUROC using scikit-learn
    auroc_softmax = roc_auc_score(gt_flat, pred_flat)

    print(f"AUROC (softmax prediction map): {auroc_softmax:.3f}")

    # Convert softmax prediction map to binary map
    pred_binary = (pred_map > 0.5).astype(int)

    # Calculate AUROC using binary prediction map
    auroc_binary = roc_auc_score(gt_map.flatten(), pred_binary.flatten())
    print(f"AUROC (binary prediction map): {auroc_binary:.3f}")
    # Calculate confusion matrices
    tn_softmax, fp_softmax, fn_softmax, tp_softmax = confusion_matrix(gt_map.flatten(), pred_map.flatten()).ravel()
    tn_binary, fp_binary, fn_binary, tp_binary = confusion_matrix(gt_map.flatten(), pred_binary.flatten()).ravel()
    # Calculate sensitivity and specificity
    sensitivity_softmax = tp_softmax / (tp_softmax + fn_softmax)
    specificity_softmax = tn_softmax / (tn_softmax + fp_softmax)

    sensitivity_binary = tp_binary / (tp_binary + fn_binary)
    specificity_binary = tn_binary / (tn_binary + fp_binary)

    print(f"Softmax Sensitivity: {sensitivity_softmax:.3f}")
    print(f"Binary Sensitivity: {sensitivity_binary:.3f}")
    print(f"Softmax Specificity: {specificity_softmax:.3f}")
    print(f"Binary Specificity: {specificity_binary:.3f}")

    # Calculate ROC curve and AUROC for softmax predictions
    fpr_softmax, tpr_softmax, _ = roc_curve(gt_map.flatten(), pred_map.flatten())
    auroc_softmax = auc(fpr_softmax, tpr_softmax)

    # Convert softmax prediction map to binary map
    pred_binary = (pred_map > 0.5).astype(int)

    # Calculate ROC curve and AUROC for binary predictions
    fpr_binary, tpr_binary, _ = roc_curve(gt_map.flatten(), pred_binary.flatten())
    auroc_binary = auc(fpr_binary, tpr_binary)

    # # Plot the ROC curves
    # plt.figure(figsize=(8, 6))
    # plt.plot(fpr_binary, tpr_binary, color='red', lw=2, label=f'Binary AUROC: {auroc_binary:.4f}')
    # plt.plot(fpr_softmax, tpr_softmax, color='blue', lw=2, label=f'Softmax AUROC: {auroc_softmax:.4f}')
    # plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC Curves')
    # plt.legend(loc="lower right")
    # plt.show()

if __name__ == '__main__':
    subject_list = [
        "prostate_004",
        "prostate_028",
        "prostate_041",
        "prostate_052",
        "prostate_069",
        "prostate_094",
        "prostate_113",
        "prostate_121",
        "prostate_145",
        "prostate_154"
    ]
    for file_name in subject_list:
        print(f'processing {file_name}')
        # y_true_path = rf"C:\Users\dzha937\DEV\RSTrial\draft_labels\{file_name}.nii.gz"
        y_true_path = rf"C:\Users\dzha937\DEV\RSTrial\cropped_labels\{file_name}.nii.gz"
        # y_pred_path = rf"C:\Users\dzha937\DEV\RSTrial\Dataset713_picai_baseline\nnunetv2-vanilla-prediction0\{file_name}.nii.gz"
        # y_pred_path = rf"C:\Users\dzha937\DEV\RSTrial\Task2203_picai_baseline\uncropped_registered_prediction\{file_name}.nii.gz"
        y_pred_path = rf"C:\Users\dzha937\DEV\RSTrial\Ella_cropped\{file_name}.nii.gz"
        cal_auroc(y_true_path, y_pred_path)