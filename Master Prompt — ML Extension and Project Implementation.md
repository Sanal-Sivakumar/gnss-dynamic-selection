# ROLE

You are the lead ML engineer and systems engineer responsible for extending an existing GNSS receiver reliability project.

You must first study the uploaded original project report and any project files/code available in the workspace. Treat the original report as the authoritative specification of the existing system.

The goal is NOT to replace the existing project with a different ML problem.

The goal is to add a machine-learning component that learns the context-dependent importance of the existing reliability parameters and dynamically determines the weights used by the original reliability equation.

Do not begin implementation until you have understood the existing architecture, equations, data flow, hardware, software, variables, and limitations.

---

# 1. ORIGINAL PROJECT — PRESERVE THIS CORE IDEA

The existing project evaluates two GNSS receivers and selects the more reliable source.

The original reliability equation is:

W = αT + βS + γSNR + δDOP

where:

- T = time/synchronization reliability
- S = satellite-count reliability
- SNR = signal-quality reliability
- DOP = geometry/reliability measure

The original implementation uses fixed weights:

α = 0.25
β = 0.25
γ = 0.30
δ = 0.20

The receiver with the higher reliability score is selected:

Best Source = argmax(W)

THIS EQUATION MUST REMAIN THE CORE DECISION EQUATION.

Do not replace it with a black-box classifier.

Do not replace it with a model that directly predicts the receiver.

Do not redesign the project as a generic GNSS error-prediction system.

The ML model must sit ABOVE the original equation and learn its weights.

---

# 2. PRIMARY ML OBJECTIVE

Build a machine-learning model that learns:

[α, β, γ, δ] = f(X)

where X represents the current GNSS conditions.

The learned weights must then be inserted into the original equation:

W_i(t) =
α(X_t)T_i +
β(X_t)S_i +
γ(X_t)SNR_i +
δ(X_t)DOP_i

for each receiver i.

Then:

Best Source = argmax(W_1, W_2)

The conceptual transformation is:

ORIGINAL:

GNSS measurements
→ fixed weights
→ reliability equation
→ receiver selection

ML-ENHANCED:

GNSS measurements
→ ML weight learner
→ dynamic α, β, γ, δ
→ original reliability equation
→ receiver selection

The original equation must remain interpretable and visible throughout the system.

---

# 3. WHAT THE ML MODEL MUST LEARN

The model should not simply learn one global set of weights.

It must learn context-dependent weighting.

For example, the model should be capable of learning that under one GNSS condition:

SNR may be highly informative,

while under another condition:

DOP or satellite count may become more informative.

The purpose of ML is therefore to discover nonlinear relationships and interactions between the GNSS indicators and the quality of the resulting receiver selection.

The model should learn:

current GNSS conditions
→ relative importance of T, S, SNR, DOP

rather than:

current GNSS conditions
→ receiver ID

---

# 4. INPUT FEATURES

Construct the ML input from the existing project variables wherever available.

At minimum, consider:

Receiver 1:
- T1
- S1
- SNR1
- DOP1

Receiver 2:
- T2
- S2
- SNR2
- DOP2

Also construct comparative features:

ΔT = T1 - T2
ΔS = S1 - S2
ΔSNR = SNR1 - SNR2
ΔDOP = DOP1 - DOP2

Where appropriate and supported by the existing project data, also consider temporal/contextual information such as:

- previous value
- rate of change
- short-term moving average
- short-term variance
- sudden changes in satellite count
- sudden changes in SNR
- sudden changes in DOP
- recent receiver-switch state

Do not add arbitrary features without explaining why they are relevant.

Do not use features that would not be available during real-time inference.

Create a clear feature table showing:

Feature
Meaning
Source
Units
Normalization
Availability during deployment

---

# 5. NORMALIZATION

Because the input variables have different numerical ranges, normalize/standardize the ML inputs.

The preprocessing procedure must be fitted using training data only.

The same fitted preprocessing parameters must then be used for:

- validation data
- test data
- real-time inference

Prevent data leakage.

Save the preprocessing parameters with the trained model.

---

# 6. ML ARCHITECTURE

Use a compact neural network suitable for deployment on the project's Raspberry Pi environment.

A suitable initial architecture is:

Input
→ Dense layer
→ ReLU
→ Dense layer
→ ReLU
→ Dense(4)
→ Softmax

The four outputs correspond to:

α
β
γ
δ

Softmax is required so that:

α ≥ 0
β ≥ 0
γ ≥ 0
δ ≥ 0

and:

α + β + γ + δ = 1

This maintains the interpretation of the outputs as relative weights.

Do not unnecessarily use a very large model.

The final model should be lightweight enough for real-time inference.

---

# 7. IMPORTANT DISTINCTION

There are TWO sets of weights in this system.

Do not confuse them.

A) Neural-network parameters

These are the internal parameters learned through backpropagation.

B) GNSS reliability weights

These are:

α, β, γ, δ

The neural network learns its internal parameters so that it can produce useful values of α, β, γ, δ.

The output weights are then passed to the original reliability equation.

Document this distinction clearly in both the code and report.

---

# 8. TRAINING OBJECTIVE

The training objective must ultimately be connected to the project's real engineering objective:

SELECT THE RECEIVER THAT PRODUCES THE BETTER POSITIONING RESULT.

For every training sample/time window, determine which receiver performed better according to the available ground-truth/reference measurement.

If receiver 1 has actual positioning error E1 and receiver 2 has actual positioning error E2:

if E1 < E2:

Receiver 1 is the preferred reference choice.

if E2 < E1:

Receiver 2 is the preferred reference choice.

The training objective should encourage the learned weights to cause the original reliability equation to select the receiver with lower actual error.

Do not train the network merely to imitate the original 0.25/0.25/0.30/0.20 weights.

That would not constitute meaningful learning.

---

# 9. DIFFERENTIABLE TRAINING

The deployed system can use:

Best Source = argmax(W1, W2)

However, direct argmax is not differentiable.

Therefore, during neural-network training, implement an appropriate differentiable/soft selection mechanism.

For example, convert W1 and W2 into soft selection probabilities using a temperature-controlled softmax:

P1 = softmax(W1, W2)
P2 = softmax(W1, W2)

Then construct a training loss related to the actual positioning errors:

Loss ≈ P1E1 + P2E2

where:

E1 = actual positioning error of receiver 1
E2 = actual positioning error of receiver 2

Investigate and justify the exact loss formulation before implementation.

The deployed system must revert to the normal deterministic selection:

if W1 > W2:
    select receiver 1
else:
    select receiver 2

---

# 10. HANDLE TIES AND NEAR-TIES

Do not switch receivers when their reliability scores are almost identical.

Implement a configurable hysteresis/deadband mechanism.

For example:

if:

|W1 - W2| < threshold

retain the currently selected receiver.

The threshold must be experimentally determined or clearly configurable.

The purpose is to prevent rapid oscillation between receivers.

Report the switching behaviour before and after hysteresis.

---

# 11. TRAINING STRATEGY

Do NOT immediately jump to the neural network.

Build the following three systems and compare them:

## Model A — Original fixed-weight system

α = 0.25
β = 0.25
γ = 0.30
δ = 0.20

This is the original baseline.

## Model B — Globally optimized fixed weights

Find one optimal set:

α*
β*
γ*
δ*

subject to:

α + β + γ + δ = 1

and:

α, β, γ, δ ≥ 0

These weights remain constant for all conditions.

This establishes whether merely optimizing the fixed coefficients provides an improvement.

## Model C — Dynamic ML weighting

Learn:

[α, β, γ, δ] = f(X)

so the weights change according to GNSS conditions.

This is the proposed ML system.

The final evaluation must compare all three.

This comparison is essential because otherwise it is impossible to determine whether the improvement comes from ML adaptivity or simply from choosing better fixed weights.

---

# 12. DATASET CONSTRUCTION

Inspect all available project data first.

Determine:

- what measurements exist
- sampling frequency
- receiver synchronization
- timestamps
- missing values
- invalid values
- positioning reference
- error calculation
- number of samples
- test scenarios
- static/dynamic measurements

Construct samples chronologically.

Avoid random row-level splitting when adjacent rows belong to the same continuous experiment/session.

Prefer experiment/session-level separation to prevent temporal leakage.

Clearly document:

Training set
Validation set
Test set

and ensure that no test information influences model training, normalization, hyperparameter selection, or threshold selection.

---

# 13. DATA INTEGRITY — CRITICAL

Never misrepresent generated or simulated data as real measurements.

If any portion of the dataset is generated, simulated, augmented, or derived artificially, label it accurately.

If actual hardware measurements exist, clearly separate:

1. Real measured GNSS data
2. Generated/simulated data
3. Derived features
4. Ground-truth/reference data

The report must never claim that generated observations are real observations.

If the project requires real-world validation, explicitly identify it as a validation requirement.

The goal is to make the methodology scientifically defensible.

---

# 14. MODEL EVALUATION

Do not evaluate the ML model only by neural-network loss.

The engineering evaluation must focus on receiver-selection performance.

Calculate, at minimum:

- selected receiver
- actual error of selected receiver
- error of receiver 1
- error of receiver 2
- selection accuracy
- mean selected error
- median selected error
- RMSE
- maximum error
- high-percentile error
- number of receiver switches
- unnecessary switches
- average duration of a selected receiver
- performance during difficult GNSS conditions

Also compare:

Original fixed-weight
vs
Optimized fixed-weight
vs
Dynamic ML weighting

Use the same held-out test data for all systems.

---

# 15. EVALUATE THE LEARNED WEIGHTS

Do not treat the four learned weights as an opaque output.

Analyze:

α over time
β over time
γ over time
δ over time

Investigate how they behave under different GNSS conditions.

For example:

- high satellite availability
- low satellite availability
- good geometry
- poor geometry
- high SNR
- low SNR
- stable timing
- unstable timing

The objective is to demonstrate that the model has learned meaningful context-dependent relationships.

Do not claim a causal relationship unless experimentally established.

---

# 16. AVOIDING OVERFITTING

Use:

- training/validation/test separation
- early stopping where appropriate
- model checkpointing
- suitable regularization if necessary
- validation-based hyperparameter selection

Monitor:

training loss
validation loss

If training improves while validation deteriorates, investigate overfitting.

Do not tune the final model using the test set.

---

# 17. ABLATION STUDY

Where feasible, perform ablation experiments.

Test:

1. T only
2. S only
3. SNR only
4. DOP only
5. T + S
6. T + S + SNR
7. all four
8. all four + comparative features
9. all four + temporal features

The purpose is to determine whether the model is genuinely benefiting from the different information sources.

Also compare:

fixed weights
vs
dynamic weights

This makes the ML contribution scientifically stronger.

---

# 18. REAL-TIME DEPLOYMENT

After training, export the smallest reliable model format suitable for Raspberry Pi inference.

The real-time pipeline should be:

GNSS Receiver 1
+
GNSS Receiver 2
↓
data acquisition
↓
feature calculation
↓
normalization using saved training parameters
↓
ML inference
↓
α, β, γ, δ
↓
original reliability equation
↓
W1, W2
↓
hysteresis/decision logic
↓
selected receiver
↓
existing project output/dashboard

Do not move the entire training process onto the Raspberry Pi.

Training occurs offline.

The Raspberry Pi performs inference.

---

# 19. FAIL-SAFE BEHAVIOUR

The ML system must not become a single point of failure.

Implement fallback behaviour.

If:

- ML model unavailable
- malformed input
- missing feature
- invalid GNSS measurement
- normalization failure
- inference failure
- abnormal output weights

then revert to a safe deterministic strategy.

The original fixed-weight reliability equation should remain available as a fallback.

Validate that:

α + β + γ + δ ≈ 1

and that all weights are valid.

---

# 20. LOGGING

For every decision, log:

timestamp
receiver 1 measurements
receiver 2 measurements
input features
predicted α
predicted β
predicted γ
predicted δ
W1
W2
selected receiver
switch reason
hysteresis state

This will make debugging and later analysis much easier.

---

# 21. PROJECT STRUCTURE

Before changing files, inspect the existing repository.

Do not arbitrarily rewrite working components.

Preserve existing:

- ESP32 firmware
- GNSS communication
- Raspberry Pi acquisition
- Flask/dashboard functionality
- existing reliability calculation
- receiver-selection logic

Add the ML functionality modularly.

Prefer a structure conceptually similar to:

data/
training/
models/
preprocessing/
inference/
evaluation/
visualization/

The exact structure must follow the existing project rather than being imposed blindly.

---

# 22. REQUIRED OUTPUTS

Produce:

1. Dataset-generation/processing pipeline if needed
2. Feature-engineering module
3. Training script
4. Validation/evaluation script
5. Model checkpoint
6. Saved preprocessing parameters
7. Inference module
8. Integration with the existing receiver-selection system
9. Comparison scripts
10. Plots/tables for evaluation
11. Configuration file for ML parameters
12. README explaining how to reproduce the entire experiment

Do not hard-code paths.

Do not hard-code learned weights.

Do not hard-code test results.

---

# 23. REPORT REVISION

The original report must be revised so that the final document presents the project as ONE coherent system.

Do NOT write the report chronologically as:

"First we did this, then we added ML, then we changed this..."

Instead, write the final report as though the complete proposed system is one unified methodology.

The narrative should flow:

Problem
→ Motivation
→ Existing reliability model
→ Need for adaptive weighting
→ Proposed ML-enhanced reliability model
→ System architecture
→ Data and feature preparation
→ ML methodology
→ Training
→ Integration
→ Evaluation
→ Results
→ Discussion
→ Limitations
→ Conclusion
→ Future work

The reader should understand the complete system from beginning to end without needing to reconstruct its development history.

---

# 24. REPORT CHANGES THAT MUST BE MADE

Mark every section of the original report that must change.

At minimum, review and revise:

## Abstract

Introduce the ML-based dynamic-weight mechanism.

Do not claim that ML replaces receiver selection.

State that ML dynamically determines the coefficients of the reliability equation.

## Introduction

Add the limitation of fixed manually selected weights.

Explain why context-dependent weighting is useful.

## Problem Statement

Update it to include adaptive weighting.

## Objectives

Add an objective such as:

"To develop a machine-learning model capable of learning context-dependent weights for the GNSS reliability equation."

Also include:

"To compare the adaptive ML weighting approach with the original fixed-weight method."

## Literature Review

Add relevant discussion of:

- machine learning for GNSS quality assessment
- adaptive weighting
- data-driven sensor reliability
- receiver/source selection
- interpretable ML
- dynamic sensor fusion/selection

Do not introduce literature that changes the project's fundamental objective.

## Existing System

Keep the original fixed-weight equation.

Present it as the baseline.

## Proposed System

Introduce:

ML feature input
→ dynamic weight prediction
→ original reliability equation
→ receiver selection

## Methodology

Add detailed sections for:

- dataset
- preprocessing
- feature engineering
- model architecture
- weight constraints
- training
- loss function
- validation
- test methodology
- real-time inference
- hysteresis
- fallback

## Mathematical Model

Retain:

W = αT + βS + γSNR + δDOP

Then introduce:

[α, β, γ, δ] = f(X)

and therefore:

W(t) =
α(X_t)T +
β(X_t)S +
γ(X_t)SNR +
δ(X_t)DOP

## Results

Add direct comparison between:

Original fixed weights
Optimized fixed weights
ML dynamic weights

## Discussion

Explain:

- what the ML model learned
- how weights vary
- whether adaptive weighting improves receiver selection
- where the model performs well
- where it fails
- whether switching stability improves

Do not overclaim.

## Limitations

Explicitly state limitations involving:

- dataset size
- ground-truth quality
- environmental coverage
- generalization
- real-time feature availability
- model dependency on training distribution

## Conclusion

Explain that the contribution is an adaptive, interpretable weighting layer built around the original reliability equation.

---

# 25. REPORT STYLE

The final report must NOT use a development diary.

Avoid:

"We first implemented..."
"Later we decided..."
"After that we added..."
"Initially our model..."
"Then we changed..."

Instead use formal research language:

"The proposed system..."
"The methodology consists of..."
"The model receives..."
"The learned weights are..."
"The reliability score is calculated..."
"The evaluation compares..."

The methodology should be linear and logical.

---

# 26. FIGURES TO CREATE

Create a clean final architecture diagram showing:

GNSS Receiver 1
GNSS Receiver 2
↓
Feature extraction
↓
ML dynamic-weight model
↓
α β γ δ
↓
Original reliability equation
↓
W1/W2
↓
hysteresis
↓
selected receiver
↓
Raspberry Pi / Flask dashboard

Also create:

- ML architecture diagram
- data-processing pipeline
- training pipeline
- real-time inference pipeline
- comparison methodology
- learned-weight visualization
- receiver-selection performance plots

---

# 27. EXPERIMENTAL COMPARISON

The final experiment must answer these questions:

RQ1:
Does the original reliability equation select the better receiver consistently?

RQ2:
Can fixed coefficients be optimized from the available data?

RQ3:
Does context-dependent ML weighting improve over the original fixed coefficients?

RQ4:
Does context-dependent weighting outperform a single globally optimized set of coefficients?

RQ5:
Are the learned weights interpretable and related to changing GNSS conditions?

RQ6:
Does the adaptive method introduce excessive receiver switching?

Do not manufacture positive results.

If ML does not outperform the baseline, report that honestly and analyze why.

---

# 28. SCIENTIFIC INTEGRITY

Never fabricate:

- measurements
- accuracy
- improvement percentages
- datasets
- experiment results
- citations
- hardware behaviour
- real-world validation

If something has not actually been measured, label it as:

- simulated
- estimated
- proposed
- expected
- future validation

If real hardware measurements are available, distinguish them clearly from generated or simulated training data.

The final report must accurately describe the origin of every dataset used.

---

# 29. FINAL IMPLEMENTATION CHECKLIST

Before declaring the project complete, verify:

[ ] Original reliability equation preserved

[ ] Original fixed-weight system implemented as baseline

[ ] Globally optimized fixed-weight baseline implemented

[ ] ML dynamic-weight model implemented

[ ] Four ML outputs correspond to α, β, γ, δ

[ ] Softmax constrains the weights

[ ] Weight sum equals 1

[ ] Feature normalization uses training data only

[ ] No train/test leakage

[ ] Training objective relates to receiver-selection quality

[ ] Validation set used for model selection

[ ] Test set remains untouched until final evaluation

[ ] Hysteresis implemented

[ ] Fallback to deterministic method implemented

[ ] Real-time inference tested

[ ] Model is lightweight enough for Raspberry Pi

[ ] Decision logging implemented

[ ] Learned weights visualized

[ ] Baseline comparisons generated

[ ] Ablation analysis performed where feasible

[ ] Report revised consistently

[ ] All data sources accurately identified

[ ] No fabricated results

---

# 30. MOST IMPORTANT DESIGN PRINCIPLE

The final system must be explainable as:

GNSS observations
↓
ML learns the context-dependent importance of reliability indicators
↓
ML outputs α, β, γ, δ
↓
Original reliability equation calculates W
↓
Higher W indicates the selected receiver
↓
Receiver is passed to the existing system

The ML model is therefore an ADAPTIVE WEIGHT LEARNER, not a replacement for the reliability equation.

Do not deviate from this architecture unless analysis demonstrates a fundamental technical problem with it.

If an alternative is proposed, explain:

1. Why the current architecture fails
2. What the alternative changes
3. Why the alternative is scientifically superior
4. What happens to the original equation
5. What additional data or hardware is required

Do not silently change the project's research question.

---

# EXECUTION ORDER

Follow this exact order:

PHASE 1 — Understand
1. Read the entire original report.
2. Inspect all project files.
3. Map the existing architecture.
4. Identify all available measurements.
5. Identify missing variables.
6. Identify how the current reliability scores T, S, SNR and DOP are calculated.

PHASE 2 — Design
7. Define the ML inputs.
8. Define the dynamic-weight output.
9. Define the training target/ground truth.
10. Define the differentiable training objective.
11. Define the neural-network architecture.
12. Define the three experimental baselines.
13. Define evaluation metrics.

PHASE 3 — Data
14. Construct the dataset.
15. Validate timestamps and receiver alignment.
16. Remove/handle invalid measurements.
17. Create features.
18. Split by experiment/session.
19. Fit preprocessing only on training data.

PHASE 4 — Training
20. Train the global fixed-weight baseline.
21. Train the ML dynamic-weight model.
22. Tune only using training/validation data.
23. Save the best model.
24. Freeze the final model.

PHASE 5 — Evaluation
25. Evaluate all approaches on untouched test data.
26. Calculate receiver-selection metrics.
27. Analyze learned weights.
28. Perform ablation studies.
29. Analyze switching behaviour.

PHASE 6 — Integration
30. Integrate inference into the existing Raspberry Pi pipeline.
31. Preserve the original equation.
32. Add hysteresis.
33. Add fallback.
34. Add logging.
35. Test real-time operation.

PHASE 7 — Report
36. Rewrite the report as one coherent final methodology.
37. Mark every modified section.
38. Add ML equations.
39. Add architecture diagrams.
40. Add experimental comparison.
41. Add limitations.
42. Ensure every data claim is truthful.
43. Ensure the final report does not read like a development diary.

Do not skip directly to coding.

First produce a concise technical assessment of the existing project and identify any missing information that prevents scientifically valid ML training. Only after that assessment should implementation begin.