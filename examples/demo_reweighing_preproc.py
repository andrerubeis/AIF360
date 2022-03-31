# %% md
#https://towardsdatascience.com/reweighing-the-adult-dataset-to-make-it-discrimination-free-44668c9379e8
#### This notebook demonstrates the use of a reweighing pre-processing algorithm for bias mitigation

# Load all necessary packages
import sys

sys.path.append("../")
import numpy as np
from tqdm import tqdm

from aif360.datasets import BinaryLabelDataset
from aif360.datasets import AdultDataset, GermanDataset, CompasDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing.reweighing import Reweighing
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions \
    import load_preproc_data_adult, load_preproc_data_german, load_preproc_data_compas
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from IPython.display import Markdown, display
import matplotlib.pyplot as plt

from common_utils import compute_metrics



#### Load dataset and set options



## import dataset
dataset_used = "adult"  # "adult", "german", "compas"
protected_attribute_used = 1  # 1, 2

if dataset_used == "adult":
    #     dataset_orig = AdultDataset()
    if protected_attribute_used == 1:
        privileged_groups = [{'sex': 1}] #males
        unprivileged_groups = [{'sex': 0}] #females
        dataset_orig = load_preproc_data_adult(['sex'])
    else:
        privileged_groups = [{'race': 1}]
        unprivileged_groups = [{'race': 0}]
        dataset_orig = load_preproc_data_adult(['race'])

elif dataset_used == "german":
    #     dataset_orig = GermanDataset()
    if protected_attribute_used == 1:
        privileged_groups = [{'sex': 1}]
        unprivileged_groups = [{'sex': 0}]
        dataset_orig = load_preproc_data_german(['sex'])
    else:
        privileged_groups = [{'age': 1}]
        unprivileged_groups = [{'age': 0}]
        dataset_orig = load_preproc_data_german(['age'])

elif dataset_used == "compas":
    #     dataset_orig = CompasDataset()
    if protected_attribute_used == 1:
        privileged_groups = [{'sex': 1}]
        unprivileged_groups = [{'sex': 0}]
        dataset_orig = load_preproc_data_compas(['sex'])
    else:
        privileged_groups = [{'race': 1}]
        unprivileged_groups = [{'race': 0}]
        dataset_orig = load_preproc_data_compas(['race'])

all_metrics = ["Statistical parity difference",
               "Average odds difference",
               "Equal opportunity difference"]

# random seed for calibrated equal odds prediction
np.random.seed(1)

# %% md

#### Split into train, and test

# %%

# Get the dataset and split into train and test
dataset_orig_train, dataset_orig_vt = dataset_orig.split([0.7], shuffle=True)
dataset_orig_valid, dataset_orig_test = dataset_orig_vt.split([0.5], shuffle=True)

# %% md

#### Clean up training data

# %%

# print out some labels, names, etc.
display(Markdown("#### Training Dataset shape"))
print(dataset_orig_train.features.shape)
display(Markdown("#### Favorable and unfavorable labels"))
print(dataset_orig_train.favorable_label, dataset_orig_train.unfavorable_label)
display(Markdown("#### Protected attribute names"))
print(dataset_orig_train.protected_attribute_names)
display(Markdown("#### Privileged and unprivileged protected attribute values"))
print(dataset_orig_train.privileged_protected_attributes,
      dataset_orig_train.unprivileged_protected_attributes)
display(Markdown("#### Dataset feature names"))
print(dataset_orig_train.feature_names)

# %% md

#### Metric for original training data

# %%

# Metric for the original dataset
metric_orig_train = BinaryLabelDatasetMetric(dataset_orig_train,
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)
display(Markdown("#### Original training dataset"))
print(
    "Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_train.mean_difference())

# %% md

#### Train with and transform the original training data

# %%

RW = Reweighing(unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups)

#store the weights based on the conditionings
RW.fit(dataset_orig_train) #        new_dataset = func(self, *args, **kwargs: returns only an object of type Reweighting

#Rebalnace the dataset with the new weights
dataset_transf_train = RW.transform(dataset_orig_train)

# %%

### Testing
assert np.abs(dataset_transf_train.instance_weights.sum() - dataset_orig_train.instance_weights.sum()) < 1e-6

# %% md

#### Metric with the transformed training data

# %%

metric_transf_train = BinaryLabelDatasetMetric(dataset_transf_train,
                                               unprivileged_groups=unprivileged_groups,
                                               privileged_groups=privileged_groups)
x = metric_transf_train.mean_difference()
display(Markdown("#### Transformed training dataset"))
print(
    "Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_transf_train.mean_difference())

# %%

### Testing
assert np.abs(metric_transf_train.mean_difference()) < 1e-6

# %% md

### Train classifier on original data

# %%

# Logistic regression classifier and predictions
scale_orig = StandardScaler()
X_train = scale_orig.fit_transform(dataset_orig_train.features)
y_train = dataset_orig_train.labels.ravel()
w_train = dataset_orig_train.instance_weights.ravel()

lmod = LogisticRegression()
lmod.fit(X_train, y_train,
         sample_weight=dataset_orig_train.instance_weights)
y_train_pred = lmod.predict(X_train)

# positive class index
pos_ind = np.where(lmod.classes_ == dataset_orig_train.favorable_label)[0][0]

dataset_orig_train_pred = dataset_orig_train.copy()
dataset_orig_train_pred.labels = y_train_pred

# %% md

#### Obtain scores for original validation and test sets

# %%

dataset_orig_valid_pred = dataset_orig_valid.copy(deepcopy=True)
X_valid = scale_orig.transform(dataset_orig_valid_pred.features)
y_valid = dataset_orig_valid_pred.labels
dataset_orig_valid_pred.scores = lmod.predict_proba(X_valid)[:, pos_ind].reshape(-1, 1)

dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
X_test = scale_orig.transform(dataset_orig_test_pred.features)
y_test = dataset_orig_test_pred.labels
dataset_orig_test_pred.scores = lmod.predict_proba(X_test)[:, pos_ind].reshape(-1, 1)

# %% md

### Find the optimal classification threshold from the validation set

# %%

num_thresh = 100
ba_arr = np.zeros(num_thresh)
class_thresh_arr = np.linspace(0.01, 0.99, num_thresh)
for idx, class_thresh in enumerate(class_thresh_arr):
    fav_inds = dataset_orig_valid_pred.scores > class_thresh
    dataset_orig_valid_pred.labels[fav_inds] = dataset_orig_valid_pred.favorable_label
    dataset_orig_valid_pred.labels[~fav_inds] = dataset_orig_valid_pred.unfavorable_label

    classified_metric_orig_valid = ClassificationMetric(dataset_orig_valid,
                                                        dataset_orig_valid_pred,
                                                        unprivileged_groups=unprivileged_groups,
                                                        privileged_groups=privileged_groups)

    ba_arr[idx] = 0.5 * (classified_metric_orig_valid.true_positive_rate() \
                         + classified_metric_orig_valid.true_negative_rate())

best_ind = np.where(ba_arr == np.max(ba_arr))[0][0]
best_class_thresh = class_thresh_arr[best_ind]

print("Best balanced accuracy (no reweighing) = %.4f" % np.max(ba_arr))
print("Optimal classification threshold (no reweighing) = %.4f" % best_class_thresh)

# %% md

### Predictions from the original test set at the optimal classification threshold

# %%

display(Markdown("#### Predictions from original testing data"))
bal_acc_arr_orig = []
disp_imp_arr_orig = []
avg_odds_diff_arr_orig = []

print("Classification threshold used = %.4f" % best_class_thresh)
for thresh in tqdm(class_thresh_arr):

    if thresh == best_class_thresh:
        disp = True
    else:
        disp = False

    fav_inds = dataset_orig_test_pred.scores > thresh
    dataset_orig_test_pred.labels[fav_inds] = dataset_orig_test_pred.favorable_label
    dataset_orig_test_pred.labels[~fav_inds] = dataset_orig_test_pred.unfavorable_label

    metric_test_bef = compute_metrics(dataset_orig_test, dataset_orig_test_pred,
                                      unprivileged_groups, privileged_groups,
                                      disp=disp)

    bal_acc_arr_orig.append(metric_test_bef["Balanced accuracy"])
    avg_odds_diff_arr_orig.append(metric_test_bef["Average odds difference"])
    disp_imp_arr_orig.append(metric_test_bef["Disparate impact"])

# %% md

#### Display results for all thresholds

# %%

fig, ax1 = plt.subplots(figsize=(10, 7))
ax1.plot(class_thresh_arr, bal_acc_arr_orig)
ax1.set_xlabel('Classification Thresholds', fontsize=16, fontweight='bold')
ax1.set_ylabel('Balanced Accuracy', color='b', fontsize=16, fontweight='bold')
ax1.xaxis.set_tick_params(labelsize=14)
ax1.yaxis.set_tick_params(labelsize=14)

ax2 = ax1.twinx()
ax2.plot(class_thresh_arr, np.abs(1.0 - np.array(disp_imp_arr_orig)), color='r')
ax2.set_ylabel('abs(1-disparate impact)', color='r', fontsize=16, fontweight='bold')
ax2.axvline(best_class_thresh, color='k', linestyle=':')
ax2.yaxis.set_tick_params(labelsize=14)
ax2.grid(True)

# %% md

# ```abs(1 - disparate
# impact)``` must
# be
# small(close
# to
# 0) for classifier predictions to be fair.
#
# However,
# for a classifier trained with original training data, at the best classification rate, this is quite high.This implies unfairness.

# %%

fig, ax1 = plt.subplots(figsize=(10, 7))
ax1.plot(class_thresh_arr, bal_acc_arr_orig)
ax1.set_xlabel('Classification Thresholds', fontsize=16, fontweight='bold')
ax1.set_ylabel('Balanced Accuracy', color='b', fontsize=16, fontweight='bold')
ax1.xaxis.set_tick_params(labelsize=14)
ax1.yaxis.set_tick_params(labelsize=14)

ax2 = ax1.twinx()
ax2.plot(class_thresh_arr, avg_odds_diff_arr_orig, color='r')
ax2.set_ylabel('avg. odds diff.', color='r', fontsize=16, fontweight='bold')
ax2.axvline(best_class_thresh, color='k', linestyle=':')
ax2.yaxis.set_tick_params(labelsize=14)
ax2.grid(True)

# %% md

# ```average
# odds
# difference = 0.5((FPR_unpriv - FPR_priv) + (TPR_unpriv - TPR_priv))
# ``` must
# be
# close
# to
# zero
# for the classifier to be fair.
#
# However,
# for a classifier trained with original training data, at the best classification rate, this is quite high.This implies unfairness.

# %% md

### Train classifier on transformed data

# %%

scale_transf = StandardScaler()
X_train = scale_transf.fit_transform(dataset_transf_train.features)
y_train = dataset_transf_train.labels.ravel()

lmod = LogisticRegression()
lmod.fit(X_train, y_train,
         sample_weight=dataset_transf_train.instance_weights)
y_train_pred = lmod.predict(X_train)

# %% md

#### Obtain scores for transformed test set

# %%

dataset_transf_test_pred = dataset_orig_test.copy(deepcopy=True)
X_test = scale_transf.fit_transform(dataset_transf_test_pred.features)
y_test = dataset_transf_test_pred.labels
dataset_transf_test_pred.scores = lmod.predict_proba(X_test)[:, pos_ind].reshape(-1, 1)

# %% md

### Predictions from the transformed test set at the optimal classification threshold

# %%

display(Markdown("#### Predictions from transformed testing data"))
bal_acc_arr_transf = []
disp_imp_arr_transf = []
avg_odds_diff_arr_transf = []

print("Classification threshold used = %.4f" % best_class_thresh)
for thresh in tqdm(class_thresh_arr):

    if thresh == best_class_thresh:
        disp = True
    else:
        disp = False

    fav_inds = dataset_transf_test_pred.scores > thresh
    dataset_transf_test_pred.labels[fav_inds] = dataset_transf_test_pred.favorable_label
    dataset_transf_test_pred.labels[~fav_inds] = dataset_transf_test_pred.unfavorable_label

    metric_test_aft = compute_metrics(dataset_orig_test, dataset_transf_test_pred,
                                      unprivileged_groups, privileged_groups,
                                      disp=disp)

    bal_acc_arr_transf.append(metric_test_aft["Balanced accuracy"])
    avg_odds_diff_arr_transf.append(metric_test_aft["Average odds difference"])
    disp_imp_arr_transf.append(metric_test_aft["Disparate impact"])

# %% md

#### Display results for all thresholds

# %%

fig, ax1 = plt.subplots(figsize=(10, 7))
ax1.plot(class_thresh_arr, bal_acc_arr_transf)
ax1.set_xlabel('Classification Thresholds', fontsize=16, fontweight='bold')
ax1.set_ylabel('Balanced Accuracy', color='b', fontsize=16, fontweight='bold')
ax1.xaxis.set_tick_params(labelsize=14)
ax1.yaxis.set_tick_params(labelsize=14)

ax2 = ax1.twinx()
ax2.plot(class_thresh_arr, np.abs(1.0 - np.array(disp_imp_arr_transf)), color='r')
ax2.set_ylabel('abs(1-disparate impact)', color='r', fontsize=16, fontweight='bold')
ax2.axvline(best_class_thresh, color='k', linestyle=':')
ax2.yaxis.set_tick_params(labelsize=14)
ax2.grid(True)

# %% md

# ```abs(1 - disparate
# impact)``` must
# be
# small(close
# to
# 0) for classifier predictions to be fair.
#
# For
# a
# classifier
# trained
# with reweighted training data, at the best classification rate, this is indeed the case.
# This
# implies
# fairness.

# %%

fig, ax1 = plt.subplots(figsize=(10, 7))
ax1.plot(class_thresh_arr, bal_acc_arr_transf)
ax1.set_xlabel('Classification Thresholds', fontsize=16, fontweight='bold')
ax1.set_ylabel('Balanced Accuracy', color='b', fontsize=16, fontweight='bold')
ax1.xaxis.set_tick_params(labelsize=14)
ax1.yaxis.set_tick_params(labelsize=14)

ax2 = ax1.twinx()
ax2.plot(class_thresh_arr, avg_odds_diff_arr_transf, color='r')
ax2.set_ylabel('avg. odds diff.', color='r', fontsize=16, fontweight='bold')
ax2.axvline(best_class_thresh, color='k', linestyle=':')
ax2.yaxis.set_tick_params(labelsize=14)
ax2.grid(True)

# %% md

# ```average
# odds
# difference = 0.5((FPR_unpriv - FPR_priv) + (TPR_unpriv - TPR_priv))
# ``` must
# be
# close
# to
# zero
# for the classifier to be fair.
#
# For
# a
# classifier
# trained
# with reweighted training data, at the best classification rate, this is indeed the case.
# This
# implies
# fairness.

# %% md

# Summary of Results
# We
# show
# the
# optimal
# classification
# thresholds, and the
# fairness and accuracy
# metrics.

# %% md

### Classification Thresholds

# | Dataset | Classification
# threshold |
# | - | - |
# | Adult | | 0.2674 |
# | German | 0.6732 |
# | Compas | 0.5148 |

# %% md

### Fairness Metric: Disparate impact, Accuracy Metric: Balanced accuracy

#### Performance

# | Dataset | Sex(Acc - Bef) | Sex(Acc - Aft) | Sex(Fair - Bef) | Sex(Fair - Aft) | Race / Age(Acc - Bef) | Race / Age(
#     Acc - Aft) | Race / Age(Fair - Bef) | Race / Age(Fair - Aft) |
# | - | - | - | - | - | - | - | - | - |
# | Adult(Test) | 0.7417 | 0.7128 | 0.2774 | 0.7625 | 0.7417 | 0.7443 | 0.4423 | 0.7430 |
# | German(Test) | 0.6524 | 0.6460 | 0.9948 | 1.0852 | 0.6524 | 0.6460 | 0.3824 | 0.5735 |
# | Compas(Test) | 0.6774 | 0.6562 | 0.6631 | 0.8342 | 0.6774 | 0.6342 | 0.6600 | 1.1062 |

# %% md

### Fairness Metric: Average odds difference, Accuracy Metric: Balanced accuracy

#### Performance

# | Dataset | Sex(Acc - Bef) | Sex(Acc - Aft) | Sex(Fair - Bef) | Sex(Fair - Aft) | Race / Age(Acc - Bef) | Race / Age(
#     Acc - Aft) | Race / Age(Fair - Bef) | Race / Age(Fair - Aft) |
# | - | - | - | - | - | - | - | - | - |
# | Adult(Test) | 0.7417 | 0.7128 | -0.3281 | -0.0266 | 0.7417 | 0.7443 | -0.1991 | -0.0395 |
# | German(Test) | 0.6524 | 0.6460 | 0.0071 | 0.0550 | 0.6524 | 0.6460 | -0.3278 | -0.1944 |
# | Compas(Test) | 0.6774 | 0.6562 | -0.2439 | -0.0946 | 0.6774 | 0.6342 | -0.1927 | 0.1042 |

# %%


