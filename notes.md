## Code Structure

### class `Metrics` has a variable named lesion_results, how is it calculated? 
- core code: `evaluate_case`
- when calculate overlap for lesions, the overlap_func can be 'IoU' for Intersection over Union, or 'DSC' for Dice similarity coefficient, or a custom fuc. Check the doc of `evaluate` below.
  - min_overlap: defines the minimal required Intersection over Union (IoU) or Dice similarity
          coefficient (DSC) between a lesion candidate and ground truth lesion, to be counted as a true
          positive detection.
  - overlap_func: function to calculate overlap between a lesion candidate and ground truth mask.
      May be 'IoU' for Intersection over Union, or 'DSC' for Dice similarity coefficient. Alternatively,
      provide a function with signature `func(detection_map, annotation) -> overlap [0, 1]`.


### How is the confidence for a lesion calculated?

- core code: `extract_lesion_candidates_dynamic`
- it seems that getting the max probability within the lesion as the confidence. Check the doc of `evaluate` below.
  - case_confidence_func: function to derive case-level confidence from detection map. Default: max.

## Definition of TPs, FNs, and FPs in PI-CAI
- all lesion candidates that are matched are TPs: 
(1, lesion_confidence, overlap)

- all ground truth lesions that are not matched are FNs:
(1, 0., 0.)

- all lesion candidates with insufficient overlap/not matched to a gt lesion are FPs:
(0, lesion_confidence, 0.)

## My Definition of TPs, FNs, FPs and TNs

min_overlap = 0.1

min_confidence = 0.5

### Group 1: all lesion candidates that are matched (`overlap > min_overlap`)
**Classification Criteria**:
   - If `lesion_confidence > min_confidence`: Classify as **TP**.
   - Otherwise: Classify as **FN**.

### Group 2: Ground Truth Lesions Not Matched
**Classification**: All are classified as **FN**.

### Group 3: all lesion candidates with insufficient overlap/not matched to a gt lesion

1. **Evaluate unmatched lesion candidates.**
2. **Classification Criteria**:
   - If `lesion_confidence > min_confidence`: Classify as **FP**.
   - If `lesion_confidence <= min_confidence`: Classify as **TN**.
