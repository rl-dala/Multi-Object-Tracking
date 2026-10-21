# Multi-Object-Tracking
Pet project on tracking objects

MOTA (Multi Object Tracking Accuracy) metric is computed across all frames as:
$$ MOTA = 1 - \frac{\sum_{t}(FN_t+FP_t+IDSW_t)}{\sum_{t}GT_t}$$
Where $t$ is the frame index, FN are false negatives, FP are false positives, IDSW are identity switches, and GT is the number of ground truth objects.

The computation of MOTA involves temporal dependency, penalizing track assignment changes between consecutive frames. An IDSW is counted when a GT target matches track $j$ in the current frame but was matched to a different track $k (k \neq j)$ in the previous frame.

Note: MOTA can be negative if there is more errors than there were actual objects to track.

IDF1 (Identity F1 score) evaluates how long the tracker correctly identifies an object, rather than just counting errors. It computes the assignment between prediction and ground truth objects across the entire video, rather than frame by frame.
$$ IDF1 = \frac{2 \times ID TP}{2 \times ID TP + ID FP + ID FN} = \frac{2 \times Precision \times Recall}{Precision + Recall}$$

To add:
* Add following metrics:
    * MOTA
    * IDF1
    * ID switches