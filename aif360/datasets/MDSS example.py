import os, sys, itertools
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics.mdss_classification_metric import MDSSClassificationMetric
from aif360.metrics.mdss.ScoringFunctions.Bernoulli import Bernoulli


from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_compas
from aif360.algorithms.inprocessing.meta_fair_classifier import MetaFairClassifier

from IPython.display import Markdown, display
import numpy as np
import pandas as pd
from aif360.metrics import BinaryLabelDatasetMetric
import numpy as np

np.random.seed(0)

dataset_orig = load_preproc_data_compas()

female_group = [{'sex': 1}]
male_group = [{'sex': 0}]
sys.path.append(os.getcwd())

dataset_orig_df = pd.DataFrame(dataset_orig.features, columns=dataset_orig.feature_names)
#print(dataset_orig_df.head())
#features:    sex  race  age_cat=25 to 45  age_cat=Greater than 45  age_cat=Less than 25  \
#             priors_count=0  priors_count=1 to 3  priors_count=More than 3  \
#             c_charge_degree=F  c_charge_degree=M

print(dataset_orig_df[['age_cat=Less than 25', 'age_cat=25 to 45', 'age_cat=Greater than 45']])
a = np.argmax(dataset_orig_df[['age_cat=Less than 25', 'age_cat=25 to 45',
                                     'age_cat=Greater than 45']])
a = np.argmax(dataset_orig_df[['age_cat=Less than 25', 'age_cat=25 to 45',
                                     'age_cat=Greater than 45']].values, axis = 1)

#1. np.argmax restituisce l'indice dell valore massimo presente nella matrice

#2. dataset_orig_df[['age_cat=Less than 25', 'age_cat=25 to 45', 'age_cat=Greater than 45']]

#   Seleziono solo le colonne "Less than 25" etc che possono assumere solo valori di 0 o 1
#   (Less than 25 = 1, la persona ha meno di 25 anni, altrimenti se 0 la persona ha più di 25 anni)

#3. dataset_orig_df[['priors_count=0', 'priors_count=1 to 3','priors_count=More than 3']].values, axis = 1)
#   Di tutto il dataset avente quelle 3 colonne seleziono solo gli elementi pari a 1 quindi siccome axis = 1
#   leggo la matrice per righe e ottengo un vettore pari al numero di righe avente in ogni cella l'indice
#   dell'elemento massimo appartenente a quella riga (teoricamente funziona anche senza mettere values)

#4. age_cat = np.argmax(dataset_orig_df[['age_cat=Less than 25', 'age_cat=25 to 45','age_cat=Greater than 45']].values, axis=1).reshape(-1, 1)

#   Mettendo 1 come secondo elemento a reshape vuol dire che voglio avere una singola colonna, dopodichè metto -1 che significa che la dimensione
#   del numero di righe viene intuita automaticamente

#   Nel nostro caso siccome abbiamo un vettore di 5728 elementi pari al numero di righe del dataset
#   dataset_orig_df[['age_cat=Less than 25', 'age_cat=25 to 45', 'age_cat=Greater than 45']] facendo reshape(-1,1) stiamo dicendo
#   di ottenere un nuovo vettore ma questa volta come vettore colonna il cui numero di righe intuito sarà uguale a 5728

#   Ripeto lo stesso per gli attributi priors e charge_degree

age_cat = np.argmax(dataset_orig_df[['age_cat=Less than 25', 'age_cat=25 to 45',
                                     'age_cat=Greater than 45']].values, axis=1).reshape(-1, 1)
priors_count = np.argmax(dataset_orig_df[['priors_count=0', 'priors_count=1 to 3',
                                          'priors_count=More than 3']].values, axis=1).reshape(-1, 1)
c_charge_degree = np.argmax(dataset_orig_df[['c_charge_degree=F', 'c_charge_degree=M']].values, axis=1).reshape(-1, 1)

#   Concateno i vettori colonna insieme in modo tale da ottenere un dataset finale 5728x6
features = np.concatenate((dataset_orig_df[['sex', 'race']].values, age_cat, priors_count, \
                           c_charge_degree, dataset_orig.labels), axis=1)
feature_names = ['sex', 'race', 'age_cat', 'priors_count', 'c_charge_degree']

df = pd.DataFrame(features, columns=feature_names + ['two_year_recid'])
df.head()

from aif360.datasets import StandardDataset
dataset = StandardDataset(df, label_name='two_year_recid', favorable_classes=[0],
                 protected_attribute_names=['sex', 'race'],
                 privileged_classes=[[1], [1]],
                 instance_weights_name=None)

#sex -> 1: male, 0: female
#race -> 1: caucasian 0: not caucasian

a = {2}
a.difference([2])

df.columns.get_loc

dataset_orig_train, dataset_orig_test = dataset.split([0.7], shuffle=True)
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

metric_train = BinaryLabelDatasetMetric(dataset_orig_train,
                             unprivileged_groups=male_group,
                             privileged_groups=female_group)

print("Train set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_train.mean_difference())
metric_test = BinaryLabelDatasetMetric(dataset_orig_test,
                             unprivileged_groups=male_group,
                             privileged_groups=female_group)
print("Test set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_test.mean_difference())

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(solver='lbfgs', C=1.0, penalty='l2')
clf.fit(dataset_orig_train.features, dataset_orig_train.labels.flatten())
dataset_bias_test_prob = clf.predict_proba(dataset_orig_test.features)[:,1]
dff = pd.DataFrame(dataset_orig_test.features, columns=dataset_orig_test.feature_names)
dff['observed'] = pd.Series(dataset_orig_test.labels.flatten(), index=dff.index)
dff['probabilities'] = pd.Series(dataset_bias_test_prob, index=dff.index)
dataset_bias_test = dataset_orig_test.copy()
dataset_bias_test.scores = dataset_bias_test_prob
dataset_bias_test.labels = dataset_orig_test.labels
mdss_classified = MDSSClassificationMetric(dataset_orig_test, dataset_bias_test,
                         unprivileged_groups=male_group,
                         privileged_groups=female_group)
female_privileged_score = mdss_classified.score_groups(privileged=True)
female_privileged_score

male_unprivileged_score = mdss_classified.score_groups(privileged=False)
male_unprivileged_score

mdss_classified = MDSSClassificationMetric(dataset_orig_test, dataset_bias_test,
                         unprivileged_groups=female_group,
                         privileged_groups=male_group)
male_privileged_score = mdss_classified.score_groups(privileged=True)
male_privileged_score
female_unprivileged_score = mdss_classified.score_groups(privileged=False)
female_unprivileged_score

privileged_subset = mdss_classified.bias_scan(penalty=0.5, privileged=True)
unprivileged_subset = mdss_classified.bias_scan(penalty=0.5, privileged=False)

print(privileged_subset)
print(unprivileged_subset)

assert privileged_subset[0]
assert unprivileged_subset[0]

protected_attr_names = set(privileged_subset[0].keys()).union(set(unprivileged_subset[0].keys()))
dataset_orig_test.protected_attribute_names = list(protected_attr_names)
dataset_bias_test.protected_attribute_names = list(protected_attr_names)

protected_attr = np.where(np.isin(dataset_orig_test.feature_names, list(protected_attr_names)))[0]

dataset_orig_test.protected_attributes = dataset_orig_test.features[:, protected_attr]
dataset_bias_test.protected_attributes = dataset_bias_test.features[:, protected_attr]

display(Markdown("#### Training Dataset shape"))
print(dataset_bias_test.features.shape)
display(Markdown("#### Favorable and unfavorable labels"))
print(dataset_bias_test.favorable_label, dataset_orig_train.unfavorable_label)
display(Markdown("#### Protected attribute names"))
print(dataset_bias_test.protected_attribute_names)
display(Markdown("#### Privileged and unprivileged protected attribute values"))
print(dataset_bias_test.privileged_protected_attributes,
      dataset_bias_test.unprivileged_protected_attributes)
display(Markdown("#### Dataset feature names"))
print(dataset_bias_test.feature_names)

# converts from dictionary of lists to list of dictionaries
a = list(privileged_subset[0].values())
subset_values = list(itertools.product(*a))

detected_privileged_groups = []
for vals in subset_values:
    detected_privileged_groups.append((dict(zip(privileged_subset[0].keys(), vals))))

a = list(unprivileged_subset[0].values())
subset_values = list(itertools.product(*a))

detected_unprivileged_groups = []
for vals in subset_values:
    detected_unprivileged_groups.append((dict(zip(unprivileged_subset[0].keys(), vals))))

metric_bias_test = BinaryLabelDatasetMetric(dataset_bias_test,
                                             unprivileged_groups=detected_unprivileged_groups,
                                             privileged_groups=detected_privileged_groups)

print("Test set: Difference in mean outcomes between unprivileged and privileged groups = %f"
      % metric_bias_test.mean_difference())

to_choose = dff[privileged_subset[0].keys()].isin(privileged_subset[0]).all(axis=1)
temp_df = dff.loc[to_choose]

"Our detected priviledged group has a size of {}, we observe {} as the average risk of recidivism, but our model predicts {}"\
.format(len(temp_df), temp_df['observed'].mean(), temp_df['probabilities'].mean())

group_obs = temp_df['observed'].mean()
group_prob = temp_df['probabilities'].mean()

odds_mul = (group_obs / (1 - group_obs)) / (group_prob /(1 - group_prob))
"This is a multiplicative increase in the odds by {}"\
.format(odds_mul)

assert odds_mul > 1

to_choose = dff[unprivileged_subset[0].keys()].isin(unprivileged_subset[0]).all(axis=1)
temp_df = dff.loc[to_choose]

"Our detected unpriviledged group has a size of {}, we observe {} as the average risk of recidivism, but our model predicts {}"\
.format(len(temp_df), temp_df['observed'].mean(), temp_df['probabilities'].mean())

group_obs = temp_df['observed'].mean()
group_prob = temp_df['probabilities'].mean()

odds_mul = (group_obs / (1 - group_obs)) / (group_prob /(1 - group_prob))
"This is a multiplicative decrease in the odds by {}"\
.format(odds_mul)

assert odds_mul < 1