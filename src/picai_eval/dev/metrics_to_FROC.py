import os
import pandas as pd
import matplotlib.pyplot as plt

from picai_eval.metrics import *

def analyze_model(metrics_json_path):
    metrics = Metrics(metrics_json_path)
    return metrics.lesion_TPR, metrics.lesion_FP / metrics.num_cases


if __name__ == '__main__':
    # === 加载两个模型的结果 ===
    file1 = r"Y:\picai\workdir\nnUNet_results\nnUNet\3d_fullres\Task2402_Z_SSMNet\nnUNetTrainerV2_CELossNonBatch_SPL_Baseline__nnUNetPlansv2.1\fold_0\validation_raw-best\metrics_DSC_0.1_300cases_full_185132.json"
    file2 = r"Y:\picai\picai_nnunet_semi_supervised_gc_algorithm\results\nnUNet\3d_fullres\Task2402_Z_SSMNet\nnUNetTrainerV2_CELossNonBatch_SPL_HardFirst__nnUNetPlansv2.1\fold_0\validation_raw-final\metrics_DSC_0.1_300cases_full_26433.json"
    # === 分析两个模型 ===
    sens1, fppi1 = analyze_model(file1)
    sens2, fppi2 = analyze_model(file2)

    # === 绘制 FROC 曲线对比 ===
    plt.figure(figsize=(8, 6))
    plt.plot(fppi1, sens1, label="Baseline", marker='o')
    plt.plot(fppi2, sens2, label="SPL-Hard", marker='s')
    plt.xlabel("False Positives per Image")
    plt.ylabel("Sensitivity")
    plt.title("FROC Curve Comparison")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    save_dir = os.path.dirname(file2)
    save_path = os.path.join(save_dir, "froc_comparison_two_models.png")
    plt.savefig(save_path)
    plt.close()
    print("FROC comparison saved to 'froc_comparison_two_models.png'")
