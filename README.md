# Periodontal Bone Loss Analysis via Keypoint Detection With Heuristic Post-Processing

This code accompanies the paper "Periodontal Bone Loss Analysis via Keypoint Detection With Heuristic Post-Processing" 


Please follow the instructions in Train_and_Eval.ipynb or follow this readme.




## Overview

This code can be used with any baseline landmark/pose estimation model, contributions are in the dental imaging specific post processing module.

### Abstract


Objectives: This study proposes a deep learning framework and annotation methodology for the automatic detection of periodontal bone loss landmarks, associated conditions, and staging.

Methods: 192 periapical radiographs were collected and annotated with a stage agnostic methodology, labelling clinically relevant landmarks regardless of disease presence or extent. We propose a heuristic post-processing module that aligns predicted keypoints to tooth boundaries using an auxiliary instance segmentation model. An evaluation metric, Percentage of Relative Correct Keypoints (P RCK), is proposed to capture keypoint performance in dental imaging domains. Four donor pose estimation models were adapted with fine-tuning for our keypoint problem.

Results: Post-processing improved fine-grained localisation, raising average P RCK0.05 by +0.028, but reduced coarse performance for P RCK0.25 by −0.0523 and P RCK0.5 by −0.0345. Orientation estimation shows excellent performance for auxiliary segmentation when filtered with either stage 1 object detection model. Periodontal staging was detected sufficiently, with the best mesial and distal Dice scores of 0.508 and 0.489, while furcation involvement and widened periodontal ligament space tasks remained challenging due to scarce positive samples. Scalability is implied with similar validation and external set performance.

Conclusion: The annotation methodology enables stage agnostic training with balanced representation across disease severities for some detection tasks. The P RCK metric provides a domain-specific alternative to generic pose metrics, while the heuristic post-processing module consistently corrected implausible predictions with occasional catastrophic failures. Clinical significance: The proposed framework demonstrates the feasibility of clinically interpretable periodontal bone loss assessment, with potential to reduce diagnostic variability and clinician workload.


### Quantitative Results

<p align="center">
	<img width=900, src="git_images/results_table_kpt.PNG"> <br />
	<em>
		Figure 1: Table containing PRCK keypoint results for all models, with and without postprocessing, for the validation and external sets, at thresholds 0.5, 0.25, and 0.05. Results are reported as mean(±standard deviation), where standard deviation is calculated over 5-folds.
	</em>
</p>


### Qualitative Results

<p align="center">
	<img width=900, src="git_images/results_qual_kpt.PNG"> <br />
	<em>
		 Figure 2: Six validation images with overlay keypoint results, where red points are the raw keypoint predictions and green points are the post-processed keypoints.
	</em>
</p>



## Usage


+ Install requirements

```

```


