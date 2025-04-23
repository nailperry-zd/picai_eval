from picai_eval import Metrics
import openpyxl
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def parse_ytrue_ypred(metrics_json_path):
    metrics = Metrics(metrics_json_path)
    lesion_results = metrics.lesion_results
    lesion_results_keys = lesion_results.keys()
    lesion_list = []
    for lesion_results_key in lesion_results_keys:
        tem_list = lesion_results[lesion_results_key]
        for lesion in tem_list:
            lesion_list.append(lesion)
    # collect targets and predictions
    y_true = np.array([target for target, *_ in lesion_list])
    y_pred = np.array([pred for _, pred, *_ in lesion_list])
    y_overlap = np.array([overlap for _, _, overlap in lesion_list])
    return y_true, y_pred, y_overlap

def compare_AUROC(metrics_json_path1, metrics_json_path2):
    # Parse the metrics
    y_true1, y_pred1, _ = parse_ytrue_ypred(metrics_json_path1)
    y_true2, y_pred2, _ = parse_ytrue_ypred(metrics_json_path2)

    # Calculate ROC curve and AUC for both models using continuous predictions
    fpr1, tpr1, thresholds1 = roc_curve(y_true1, y_pred1)
    roc_auc1 = auc(fpr1, tpr1)

    fpr2, tpr2, thresholds2 = roc_curve(y_true2, y_pred2)
    roc_auc2 = auc(fpr2, tpr2)

    print(f"thresholds1={thresholds1}")
    print(f"thresholds2={thresholds2}")

    # Plotting the ROC curves
    plt.figure()
    plt.plot(fpr1, tpr1, color='blue', lw=2, label=f'Model 1 (AUC = {roc_auc1:.2f})')
    plt.plot(fpr2, tpr2, color='green', lw=2, label=f'Model 2 (AUC = {roc_auc2:.2f})')

    # Plotting the diagonal line for random guess
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--')

    # Adding labels and title
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid()

    # Show the plot
    plt.show()

def calculate_mean_overlap(y_true, y_pred, y_overlap, threshold):
    # Identify true positives based on the current threshold
    true_positives_indices = np.where((y_true == 1) & (y_pred >= threshold))

    # Extract overlaps for true positives
    true_positive_overlaps = y_overlap[true_positives_indices]

    # Calculate mean overlap, handling the case where there are no true positives
    mean_overlap = np.mean(true_positive_overlaps) if true_positive_overlaps.size > 0 else 0

    return mean_overlap

def compare_Overlap_TPs_LineFigure(metrics_json_path1, metrics_json_path2):
    # Parse the metrics
    y_true1, y_pred1, y_overlap1 = parse_ytrue_ypred(metrics_json_path1)
    y_true2, y_pred2, y_overlap2 = parse_ytrue_ypred(metrics_json_path2)

    # Define discrete thresholds from 0.1 to 1.0
    thresholds = np.arange(0.1, 1.1, 0.1)
    mean_overlaps_model1 = []
    mean_overlaps_model2 = []

    # Calculate mean overlap for each threshold
    for threshold in thresholds:
        mean_overlap1 = calculate_mean_overlap(y_true1, y_pred1, y_overlap1, threshold)
        mean_overlap2 = calculate_mean_overlap(y_true2, y_pred2, y_overlap2, threshold)

        mean_overlaps_model1.append(mean_overlap1)
        mean_overlaps_model2.append(mean_overlap2)

    # Convert to numpy arrays for easier handling
    mean_overlaps_model1 = np.array(mean_overlaps_model1)
    mean_overlaps_model2 = np.array(mean_overlaps_model2)

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, mean_overlaps_model1, marker='o', label='Model 1', color='blue')
    plt.plot(thresholds, mean_overlaps_model2, marker='o', label='Model 2', color='green')

    # Adding labels and title
    plt.title('Mean Overlap of True Positives at Different Thresholds')
    plt.xlabel('Threshold')
    plt.ylabel('Mean Overlap')
    plt.xticks(thresholds)  # Set x-ticks to be the thresholds
    plt.ylim(0, 1)  # Assuming overlap is between 0 and 1
    plt.grid()
    plt.legend()

    # Show the plot
    plt.show()

def get_overlaps_for_threshold(y_true, y_pred, y_overlap, threshold):
    # Get overlap values for true positives at the current threshold
    true_positives_indices = np.where((y_true == 1) & (y_pred >= threshold))
    return y_overlap[true_positives_indices], np.sum(y_true == 1)  # Return overlaps and total positives

def compare_Overlap_TPs_Plotbox(metrics_json_path1, metrics_json_path2):
    # Parse the metrics
    y_true1, y_pred1, y_overlap1 = parse_ytrue_ypred(metrics_json_path1)
    y_true2, y_pred2, y_overlap2 = parse_ytrue_ypred(metrics_json_path2)

    # Define selected thresholds for subplots
    selected_thresholds = [0.1, 0.3, 0.5, 0.7]  # Updated thresholds
    num_thresholds = len(selected_thresholds)

    # Create subplots in a 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()  # Flatten the 2D array of axes for easy indexing

    # Plotting box plots for each selected threshold
    for i, threshold in enumerate(selected_thresholds):
        overlaps1, total_positives1 = get_overlaps_for_threshold(y_true1, y_pred1, y_overlap1, threshold)
        overlaps2, total_positives2 = get_overlaps_for_threshold(y_true2, y_pred2, y_overlap2, threshold)

        # Create box plot for the current threshold
        axes[i].boxplot([overlaps1, overlaps2], positions=[1, 2], widths=0.3)

        # Overlay individual points
        axes[i].scatter(np.ones(len(overlaps1)), overlaps1, color='blue', alpha=0.6, label='Model 1 Points')
        axes[i].scatter(np.ones(len(overlaps2)) * 2, overlaps2, color='green', alpha=0.6, label='Model 2 Points')

        # Annotate with TP count and total positives
        tp_count1 = len(overlaps1)
        tp_count2 = len(overlaps2)
        axes[i].text(1, np.max(overlaps1) + 0.02, f'TP: {tp_count1}/{total_positives1}', ha='center', color='blue')
        axes[i].text(2, np.max(overlaps2) + 0.02, f'TP: {tp_count2}/{total_positives2}', ha='center', color='green')

        # Set labels and title
        axes[i].set_xticks([1, 2])
        axes[i].set_xticklabels(['Model 1', 'Model 2'])
        axes[i].set_title(f'Threshold: {threshold}')
        axes[i].set_ylabel('Overlap Values')
        # axes[i].grid()

        # Set y-axis limits to cover 0.1 to 1.0
        axes[i].set_ylim(0.1, 1.0)  # Adjust y-axis limits

    # Set the overall title for the figure
    plt.suptitle('Overlap Values Distribution for True Positives at Selected Thresholds', fontsize=16)

    # Show the plot
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for the title
    plt.show()

def compare_AUROC_fixed_thresholds(metrics_json_path1, metrics_json_path2, common_thresholds):
    # Parse the metrics
    y_true1, y_pred1, _ = parse_ytrue_ypred(metrics_json_path1)
    y_true2, y_pred2, _ = parse_ytrue_ypred(metrics_json_path2)

    # Initialize lists to store TPR and FPR
    tpr1_list = []
    fpr1_list = []
    tpr2_list = []
    fpr2_list = []

    # Compute TPR and FPR manually for each threshold
    for threshold in common_thresholds:
        # Convert predicted probabilities to binary predictions
        y_pred1_binary = (y_pred1 >= threshold).astype(int)
        y_pred2_binary = (y_pred2 >= threshold).astype(int)

        # Get confusion matrix values
        tn1, fp1, fn1, tp1 = confusion_matrix(y_true1, y_pred1_binary).ravel()
        tn2, fp2, fn2, tp2 = confusion_matrix(y_true2, y_pred2_binary).ravel()

        # Avoid division by zero
        tpr1 = tp1 / (tp1 + fn1) if (tp1 + fn1) != 0 else 0
        fpr1 = fp1 / (fp1 + tn1) if (fp1 + tn1) != 0 else 0
        tpr2 = tp2 / (tp2 + fn2) if (tp2 + fn2) != 0 else 0
        fpr2 = fp2 / (fp2 + tn2) if (fp2 + tn2) != 0 else 0

        # Append to the lists
        tpr1_list.append(tpr1)
        fpr1_list.append(fpr1)
        tpr2_list.append(tpr2)
        fpr2_list.append(fpr2)

    # Sort points by FPR to compute AUC correctly
    fpr1_list, tpr1_list = zip(*sorted(zip(fpr1_list, tpr1_list)))
    fpr2_list, tpr2_list = zip(*sorted(zip(fpr2_list, tpr2_list)))

    # Add (0,0) and (1,1) to ensure closed ROC curve
    fpr1_list = [0.0] + list(fpr1_list) + [1.0]
    tpr1_list = [0.0] + list(tpr1_list) + [1.0]
    fpr2_list = [0.0] + list(fpr2_list) + [1.0]
    tpr2_list = [0.0] + list(tpr2_list) + [1.0]

    # Calculate AUC for both models
    roc_auc1 = auc(fpr1_list, tpr1_list)
    roc_auc2 = auc(fpr2_list, tpr2_list)

    # Plotting the ROC curves
    plt.figure()
    plt.plot(fpr1_list, tpr1_list, color='blue', lw=2, marker='o',
             label=f'Model 1 (AUC = {roc_auc1:.2f})')
    plt.plot(fpr2_list, tpr2_list, color='green', lw=2, marker='s',
             label=f'Model 2 (AUC = {roc_auc2:.2f})')

    # Plot the diagonal for random chance
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--')

    # Configure plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve at Fixed Thresholds')
    plt.legend(loc='lower right')
    plt.grid(True)

    # Show the plot
    plt.show()

if __name__ == '__main__':
    metrics_json_path1 = r"Y:\rstrial\input\images\batch1_noncropped_highb0002_registered_manual\Dataset302_rstrial_batch1\prediction0_Dataset713_picai_baseline_nnUNetTrainerFocalLoss\metrics_DSC_0.1_10cases_full_980044.json"
    metrics_json_path2 = r"Y:\rstrial\input\images\batch1_noncropped_highb0002_registered_manual\Dataset302_rstrial_batch1\prediction0_Dataset713_picai_baseline_nnUNetTrainer_new\metrics_DSC_0.1_10cases_full_318468.json"

    compare_AUROC(metrics_json_path1, metrics_json_path2)
    compare_Overlap_TPs_Plotbox(metrics_json_path1, metrics_json_path2)
    # metrics_json_dir = r"C:\Users\dzha937\DEV\RSTrial\Dataset713_picai_baseline\nnunetv2-vanilla-prediction0"
    # metrics_json_dir = r"C:\Users\dzha937\DEV\RSTrial\Ella_cropped"
    # metrics_json_dir = r"Y:\rstrial\input\images\batch1_noncropped_highb0002_registered_manual\Dataset302_rstrial_batch1\prediction0_Dataset713_picai_baseline_nnUNetTrainerFocalLoss"
    # metrics_json_dir = r"C:\Users\dzha937\DEV\PICAI\Dataset713_picai_baseline\val"
    # metrics_json_path = rf"{metrics_json_dir}\metrics_DSC_0.1_10cases_full_980044.json"
    # metrics_json_path = rf"{metrics_json_dir}\metrics_DSC_0.1_10cases_full_980044.json"
    # metrics_json_path = rf"{metrics_json_dir}\metrics_DSC_509849.json"
    # metrics_json_path = rf"{metrics_json_dir}\metrics_IoU_270247.json"
    # metrics_json_excel_path = rf"{metrics_json_dir}\metrics_DSC_0.1_10cases_full_15939.json.xlsx"
    # metrics = Metrics(metrics_json_path)
    # print(f'lesion-level metrics.AP={metrics.AP}')
    # print(f'patient-level metrics.auroc={metrics.auroc}')

    # lesion_results = metrics.lesion_results

    # lesion_results_keys = lesion_results.keys()
    # lesion_index = 1
    # lesion_list_raw = []
    # lesion_list = []
    # for lesion_results_key in lesion_results_keys:
    #     tem_list = lesion_results[lesion_results_key]
    #     for lesion in tem_list:
    #         lesion_list_raw.append(lesion)
    #         lesion_entry = dict()
    #         # if lesion[0] == 1:
    #         lesion_entry['index'] = lesion_index
    #         lesion_entry['patient_id'] = lesion_results_key
    #         lesion_entry['overlap'] = lesion[2]
    #         lesion_entry['confidence'] = lesion[1]
    #         lesion_entry['pred'] = 1 if lesion[1] > 0.5 else 0
    #         lesion_entry['label'] = lesion[0]
    #         print(f'lesion {lesion_index}, {lesion_results_key}, overlap: {lesion[2]}, confidence: {lesion[1]}')
    #         lesion_list.append(lesion_entry)
    #         lesion_index = lesion_index + 1

    # collect targets and predictions
    # y_true = np.array([target for target, *_ in lesion_list_raw])
    # y_pred = np.array([pred for _, pred, *_ in lesion_list_raw])
    # # Set a threshold to convert probabilities to binary predictions
    # threshold = 0.5
    # y_pred = (y_pred > threshold).astype(int)  # Convert to binary
    #
    # # Calculate confusion matrix
    # cm = confusion_matrix(y_true, y_pred)
    # TN, FP, FN, TP = cm.ravel()
    #
    # # Calculate sensitivity and specificity
    # sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    # specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    #
    # # Print the confusion matrix and metrics
    # print("Confusion Matrix:")
    # print(cm)
    # print(f"Sensitivity: {sensitivity:.2f}")
    # print(f"Specificity: {specificity:.2f}")
    #
    #
    #
    # # Create a new workbook
    # workbook = openpyxl.Workbook()
    #
    # # Get the active worksheet
    # worksheet = workbook.active
    #
    # # Write the header row
    # worksheet['A1'] = 'Index'
    # worksheet['B1'] = 'Patient ID'
    # worksheet['C1'] = 'Overlap'
    # worksheet['D1'] = 'Confidence'
    # worksheet['E1'] = 'Prediction'
    # worksheet['F1'] = 'Label'
    #
    # # Write the data to the worksheet
    # for row_index, entry in enumerate(lesion_list, start=2):
    #     for col_index, (item, value) in enumerate(entry.items(), start=1):
    #         worksheet.cell(row=row_index, column=col_index, value=value)
    #
    # # Save the workbook to a file
    # workbook.save(metrics_json_excel_path)


