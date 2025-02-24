# Original work Copyright 2017 Flavio Calmon
# Modified work Copyright 2018 IBM Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
import numpy as np
import pandas as pd
import cvxpy as cp
from cvxpy import Problem, Minimize, Variable


class OptTools():
    """Class that implements the optimization for optimized pre-processing.

    Based on:
    http://papers.nips.cc/paper/6988-optimized-pre-processing-for-discrimination-prevention

    and

    https://github.com/fair-preprocessing/nips2017

    The particular formulation implemented here is:
    1. l1 distance between input and transformed distributions
    2. "Excess distortion constraint" - eqn 5 in paper.
    3. Discrimination constraints for all combinations of groups specified
       (there is no distinction between protected and unprotected groups). The
       constraints are given in eqn 2, 3 in the paper. We use a single epsilon
       value for all combinations of y and d values

    See section 4.3 in supplementary material of the paper for an example

    Attributes:
        features (list): All features
        df (DataFrame): Input data
        dfJoint (DataFrame): Empirical joint distribution
        D_features (list): protected attribute names
        X_features (list): feature names for input data
        Y_features (list): feature names for binary label
        X_values (list): Values that features can take
        Y_values (list): Values that the label can take
        D_values (list): Values that protected attributes can take
        XY_features (list): Combination of X, and Y features
        DXY_features (list): Combination of D, X, and Y features
        XY_values (list): Combination of X, and Y values
        DXY_values (list): Combination of D, X, and Y values
        y_index (Int64Index): Indices for the Y values
        XY_index (MultiIndex): Indices for the combination of X, and Y values
        DXY_index (MultiIndex): Indices for the combination of D, X, and Y
            values
        YD_features_index (MultiIndex): Indices for the combination of Y, and D
            values

        clist (list): Distance thresholds for individual distortion
        CMlist (list): List of constraint matrices corresponding to each
            threshold in clist
        dfD (DataFrame): distortion matrix with indices and columns
        dlist (list): Probability bounds given in eq. 5 of the paper for
            each threshold in clist
        epsilon (float): epsilon value used in discrimination constraint

        dfD_to_Y_address (Dataframe): matrix for p_yd, with y varying in the
            columns
        dfMask_Pxyd_to_Pyd (DataFrame): Mask to transform P_XYD to P_YD
        dfMask_Pxyd_to_Pxy (DataFrame): Mask to convert from P_XYD to P_XY
        dfPxyd (DataFrame): Representation of only frequencies from dfJoint
        dfMask_Pxyd_to_Py (DataFrame): Mask to convert from P_XYD to P_Y
        dfMask_Pxy_to_Py (DataFrame): Mask to convert from P_XY to P_Y
        dfMask_Pxyd_to_Pd (DataFrame): Mask to convert from P_XYD to P_D
        dfP (DataFrame): Mapping transformation learned from the data
    """

    def __init__(self, df=None, features=None):
        """Initialize the problem. Not all attributes are initialized when
        creating the object.

        Args:
            df (DataFrame): Input dataframe
            features (list): Optional features to subset the dataframe
        """

        self.df = df.copy()

        if not isinstance(df, pd.DataFrame):
            raise TypeError("`df` must be a pandas DataFrame")

        if not features:
            self.features = list(df)
        else:
            self.features = features

        # build joint distribution : compute the probability of each possible combination in the dataset
        # from:
        #
        # race Age (decade) Education Years Income Binary sex
        #   0   10                10              <=50K     F
        #   ...
        #   1   >=70            >12              >=50K      M

        self.dfJoint = self.df.groupby(self.features).size().reset_index()
        self.dfJoint.rename(columns={0: 'Count'}, inplace=True)
        self.dfJoint['Frequency'] = self.dfJoint['Count'].apply(
            lambda x: x/float(len(self.df)))

        # initialize the features that will be used for optimization
        self.D_features = []    # discriminatory features
        self.Y_features = []    # binary decision variable
        self.X_features = []    # variables used for decision making

        # values that each feature can assume
        self.D_values = []
        self.Y_values = []

        # place holder for mapping dataframe
        self.dfP = pd.DataFrame() # this will hold the conditional mappings

        # place holder for the distortion mapping
        self.dfD = pd.DataFrame()

        # excess distortion constraint placeholder
        self.clist = []

        # excess distortion matrices
        self.CMlist = []

    def get_mask(self, dfRef):
        """Create a mask assuming the multindex column is a subset of the
        multindex rows. This mask will be used for marginalizing distributions.

        Args:
            dfRef (DataFrame): Reference data frame
        """

        # generates a mask assuming the multindex column is a subset of the
        # multindex rows

        # dfRef.index: all the combinations ('Female', 0.0,   '10',  '10', '<=50K'),
        #             ('Female', 0.0,   '10',  '10',  '>50K'), ...

        # pd.DataFrame(index=dfRef.index): empty dataframe with index values Index: [(Female, 0.0, 10, 10, <=50K), (Female, 0.0, 10, 10, >50K), (Female, 0.0, 10, 11, <=50K), (Female, 0.0, 10, 11, >50K), (Female, 0.0, 10, 12, <=50K), (Female, 0.0, 10, 12, >50K), (Female, 0.0, 10, 6, <=50K), (Female, 0.0, 10, 6, >50K), (Female, 0.0, 10, 7, <=50K), (Female, 0.0, 10, 7, >50K), (Female, 0.0, 10, 8, <=50K), (Female, 0.0, 10, 8, >50K), (Female, 0.0, 10, 9, <=50K), (Female, 0.0, 10, 9, >50K), (Female, 0.0, 10, <6, <=50K), (Female, 0.0, 10, <6, >50K), (Female, 0.0, 10, >12, <=50K), (Female, 0.0, 10, >12, >50K), (Female, 0.0, 20, 10, <=50K), (Female, 0.0, 20, 10, >50K), (Female, 0.0, 20, 11, <=50K), (Female, 0.0, 20, 11, >50K), (Female, 0.0, 20, 12, <=50K), (Female, 0.0, 20, 12, >50K), (Female, 0.0, 20, 6, <=50K), (Female, 0.0, 20, 6, >50K), (Female, 0.0, 20, 7, <=50K), (Female, 0.0, 20, 7, >50K), (Female, 0.0, 20, 8, <=50K), (Female, 0.0, 20, 8, >50K), (Female, 0.0, 20, 9, <=50K), (Female, 0.0, 20, 9, >50K), (Female, 0.0, 20, <6, <=50K), (Female, 0.0, 20, <6, >50K), (Female, 0.0, 20, >12, <=50K), (Female, 0.0, 20, >12, >50K), (Female, 0.0, 30, 10, <=50K), (Female, 0.0, 30, 10, >50K), (Female, 0.0, 30, 11, <=50K), (Female, 0.0, 30, 11, >50K), (Female, 0.0, 30, 12, <=50K), (Female, 0.0, 30, 12, >50K), (Female, 0.0, 30, 6, <=50K), (Female, 0.0, 30, 6, >50K), (Female, 0.0, 30, 7, <=50K), (Female, 0.0, 30, 7, >50K), (Female, 0.0, 30, 8, <=50K), (Female, 0.0, 30, 8, >50K), (Female, 0.0, 30, 9, <=50K), (Female, 0.0, 30, 9, >50K), (Female, 0.0, 30, <6, <=50K), (Female, 0.0, 30, <6, >50K), (Female, 0.0, 30, >12, <=50K), (Female, 0.0, 30, >12, >50K), (Female, 0.0, 40, 10, <=50K), (Female, 0.0, 40, 10, >50K), (Female, 0.0, 40, 11, <=50K), (Female, 0.0, 40, 11, >50K), (Female, 0.0, 40, 12, <=50K), (Female, 0.0, 40, 12, >50K), (Female, 0.0, 40, 6, <=50K), (Female, 0.0, 40, 6, >50K), (Female, 0.0, 40, 7, <=50K), (Female, 0.0, 40, 7, >50K), (Female, 0.0, 40, 8, <=50K), (Female, 0.0, 40, 8, >50K), (Female, 0.0, 40, 9, <=50K), (Female, 0.0, 40, 9, >50K), (Female, 0.0, 40, <6, <=50K), (Female, 0.0, 40, <6, >50K), (Female, 0.0, 40, >12, <=50K), (Female, 0.0, 40, >12, >50K), (Female, 0.0, 50, 10, <=50K), (Female, 0.0, 50, 10, >50K), (Female, 0.0, 50, 11, <=50K), (Female, 0.0, 50, 11, >50K), (Female, 0.0, 50, 12, <=50K), (Female, 0.0, 50, 12, >50K), (Female, 0.0, 50, 6, <=50K), (Female, 0.0, 50, 6, >50K), (Female, 0.0, 50, 7, <=50K), (Female, 0.0, 50, 7, >50K), (Female, 0.0, 50, 8, <=50K), (Female, 0.0, 50, 8, >50K), (Female, 0.0, 50, 9, <=50K), (Female, 0.0, 50, 9, >50K), (Female, 0.0, 50, <6, <=50K), (Female, 0.0, 50, <6, >50K), (Female, 0.0, 50, >12, <=50K), (Female, 0.0, 50, >12, >50K), (Female, 0.0, 60, 10, <=50K), (Female, 0.0, 60, 10, >50K), (Female, 0.0, 60, 11, <=50K), (Female, 0.0, 60, 11, >50K), (Female, 0.0, 60, 12, <=50K), (Female, 0.0, 60, 12, >50K), (Female, 0.0, 60, 6, <=50K), (Female, 0.0, 60, 6, >50K), (Female, 0.0, 60, 7, <=50K), (Female, 0.0, 60, 7, >50K), ...]
        # pd.DataFrame(index=dfRef.index).reset_index(): takes all the combinations and creates a dataframw in which each row correspond to a different combination
        # pd.DataFrame(index=dfRef.index).reset_index()[target_ix].values: takes all the combinations of ['race', 'Age (decade)', 'Education Years', 'Income Binary']
        target_ix = list(dfRef.columns.names) #columns names without protected attribute (['race', 'Age (decade)', 'Education Years', 'Income Binary'])
        dfRows = pd.DataFrame(index=dfRef.index).reset_index()[target_ix].values

        #dfRef.columns: names=['race', 'Age (decade)', 'Education Years', 'Income Binary']
        dfCols = pd.DataFrame(index=dfRef.columns).reset_index()[target_ix].values

        for i in range(dfRef.shape[0]):
            val1 = dfRows[i, :]
            for j in range(dfRef.shape[1]):
                val2 = dfCols[j, :]
                if np.all(val1 == val2):
                    dfRef.iat[i, j] = 1.0

        return dfRef

    # method for setting the features
    def set_features(self, D=[], X=[], Y=[]):
        """Set many features for the class

        Args:
            D (list): names of D features
            X (list): names of X features
            Y (list): names of Y features
        """

        self.D_features = D #sex
        self.Y_features = Y #income
        self.X_features = X #race, age(decade), educational years

        # Get values for Pandas multindex

        #Create lists of lists made by values composing the dataframe
        self.D_values = [self.dfJoint[feature].unique().tolist()
                         for feature in self.D_features] #Female, Male
        self.Y_values = [self.dfJoint[feature].unique().tolist()
                         for feature in self.Y_features] #<= 50K, >50K
        self.X_values = [self.dfJoint[feature].unique().tolist()
                         for feature in self.X_features]
        #[[0.0, 1.0], (race)
        #['10', '20', '30', '40', '50', '60', '>=70'], (age decade)
        #['10', '11', '12', '6', '7', '8', '9', '<6', '>12']] (educational years)

        # Create multindex for mapping dataframe

        #MultiIndex store all the combinations of mappings so we have something like:

        #Female, 0.0, 10, 10, <=50K (sex, race, age decade, educational years, income)
        #Female, 0.0, 10, 10 > 50K
        #...
        #504 total unique combinations

        self.DXY_features = self.D_features+self.X_features+self.Y_features #['sex', 'race', 'Age (decade)', 'Education Years', 'Income Binary']
        self.DXY_values = self.D_values+self.X_values+self.Y_values
        self.DXY_index = pd.MultiIndex.from_product(self.DXY_values,
                                                    names=self.DXY_features)

        # Create multindex for distortion dataframe
        self.XY_features = self.X_features+self.Y_features #all features except 'sex'
        self.XY_values = self.X_values+self.Y_values
        self.XY_index = pd.MultiIndex.from_product(self.XY_values,
                                                   names=self.XY_features)

        # Initialize mapping dataframe
        # self.dfP dataframe will be a dataset n_elements_DXY*n_elements_XY with columns all possible combinations of mappings, last columns have
        # None name
        #self.dfP: [504 rows (tot combinations of DXY) x 252 columns (tot combinations of XY)]
        self.dfP = pd.DataFrame(np.zeros((len(self.DXY_index),
                                          len(self.XY_index))),
                                index=self.DXY_index, columns=self.XY_index)

        # Initialize distortion dataframe
        # self.dfD: [252 rows (tot combinations of XY) x 252 columns (tot combinations of XY)]
        self.dfD = pd.DataFrame(np.zeros((len(self.XY_index),
                                          len(self.XY_index))),
                                index=self.XY_index.copy(),
                                columns=self.XY_index.copy())

        ###
        # Generate masks for recovering marginals pxyd
        ###
        self.dfPxyd = pd.DataFrame(index=self.dfP.index, columns=['Frequency'])
        index_list = [list(x) for x in self.dfPxyd.index.tolist()] #[['Female', 0.0, '10', '10', '<=50K'], ['Female', 0.0, '10', '10', '>50K'], ['Female', 0.0, '10', '11', '<=50K'],

        # find corresponding frequency value for each combination inside the dataset
        i = 0

        #self.dfJoint: 504 rows (total combinations) x 7 columns (race, Age (decade), Education Years, Income Binary, sex, Count, Frequency)
        #it contains the count and frequency of each combination present in the original dataset

        #self.dfPxyd: 504 (total combinations) x 1 column (it actually corresponds to the Frequency)
        for comb in self.dfJoint[self.DXY_features].values.tolist():
            # get the entry corresponding to the combination
            idx = index_list.index(comb)
            # add marginal to list
            self.dfPxyd.iloc[idx, 0] = self.dfJoint.loc[i, 'Frequency'] #get the frequency of the combination of index = idx
            i += 1

        # create mask that reduces Pxyd to Pxy
        # so Pxyd.dot(dfMask1) = Pxy
        self.dfMask_Pxyd_to_Pxy = pd.DataFrame(np.zeros((len(self.dfP),
                                                         len(self.dfD))),
                                               index=self.dfP.index,
                                               columns=self.dfD.index)
        #self.dfMask_Pxyd_to_Pxy: [504 rows x 252 columns]
        self.dfMask_Pxyd_to_Pxy = self.get_mask(self.dfMask_Pxyd_to_Pxy)

        # compute mask that reduces Pxyd to Pyd
        self.YD_features_index = self.dfJoint.groupby(
            self.Y_features+self.D_features)['Frequency'].sum().index
        self.dfMask_Pxyd_to_Pyd = pd.DataFrame(
            np.zeros((len(self.dfP), len(self.YD_features_index))),
            index=self.dfP.index, columns=self.YD_features_index)
        self.dfMask_Pxyd_to_Pyd = self.get_mask(self.dfMask_Pxyd_to_Pyd)

        # get  matrix for p_yd, with y varying in the columns
        self.dfD_to_Y_address = pd.Series(
            range(len(list(self.dfMask_Pxyd_to_Pyd))),
            index=self.dfMask_Pxyd_to_Pyd.columns)
        # print(self.dfD_to_Y_address, self.dfD_to_Y_address.shape)
        self.dfD_to_Y_address = pd.pivot_table(
            self.dfD_to_Y_address.reset_index(), columns=self.D_features,
            index=self.Y_features, values=0)

        # compute mask that reduces Pxyd to Py
        self.y_index = self.dfD_to_Y_address.index
        self.dfMask_Pxyd_to_Py = pd.DataFrame(np.zeros((len(self.dfP),
                                                        len(self.y_index))),
                                              index=self.dfP.index,
                                              columns=self.y_index)
        self.dfMask_Pxyd_to_Py = self.get_mask(self.dfMask_Pxyd_to_Py)

        # compute mask that reduces Pxy to Py
        self.dfMask_Pxy_to_Py = pd.DataFrame(np.zeros((len(list(self.dfP)),
                                                       len(self.y_index))),
                                             index=self.dfP.columns,
                                             columns=self.y_index)
        self.dfMask_Pxy_to_Py = self.get_mask(self.dfMask_Pxy_to_Py)

        # compute mask that reduces Pxyd to Pd
        self.dfMask_Pxyd_to_Pd = pd.DataFrame(
            np.zeros((len(self.dfP), self.dfD_to_Y_address.shape[1])),
            index=self.dfP.index, columns=self.dfD_to_Y_address.columns)
        self.dfMask_Pxyd_to_Pd = self.get_mask(self.dfMask_Pxyd_to_Pd)

    def set_distortion(self, get_distortion, clist=[]):
        """Create distortion and constraint matrices
        Args:
            get_distortion (function): Distortion function name
                (See optim_preproc_helper.get_distortion for an example)
            clist (list): Distance thresholds for individual distortion
        """
        #clist = [0.99, 1.99, 2.99]
        # set constraint list
        self.clist = clist

        # create row dictionay (rows represent old values)
        # this will make it easier to compute distrotion metric
        rows_tuple = self.dfD.index.tolist()
        rows_dict = [{self.XY_features[i]:t[i]
                     for i in range(len(self.XY_features))} for t in rows_tuple]

        # create columns dictionay (columns represent new values)
        cols_tuple = self.dfD.columns.tolist()
        cols_dict = [{self.XY_features[i]:t[i]
                     for i in range(len(self.XY_features))} for t in cols_tuple]

        # Create distortion matrix
        for i in range(self.dfD.shape[0]):
            old_values = rows_dict[i]
            for j in range(self.dfD.shape[1]):
                new_values = cols_dict[j]
                self.dfD.iat[i, j] = get_distortion(old_values, new_values)

        Dmatrix = self.dfD.values

        # Create constraint matrix list for excess distortion
        # since old values index the rows, we go through the D matrix line by
        # line, marking as 1 events where the threshold is violated. This will
        # be multiplied by the probability matrix, resulting in the excess
        # distortion metric
        self.CMlist = [np.zeros(Dmatrix.shape) for i in range(len(self.clist))]
        for x in range(len(self.CMlist)):
            c = self.clist[x]
            for i in range(Dmatrix.shape[0]):
                for j in range(Dmatrix.shape[1]):
                    if Dmatrix[i, j] >= c:
                        self.CMlist[x][i, j] = 1.0

    def optimize(self, epsilon=1., dlist=[], verbose=True):
        """Main optimization routine to estimate the pre-processing
        transformation.

        The particular formulation implemented here is:
        1. l1 distance between input and transformed distributions
        2. "Excess distortion constraint" - eqn 5 in paper.
        3. Discrimination constraints for all combinations of groups specified
           (there is no distinction between protected and unprotected groups).
           The constraints are given in eqn 2, 3 in the paper. We use a single
           /\epsilon value for all combinations of y and d values

        See section 4.3 in supplementary material of the paper for an example

        Args:
            epsilon (float): Distance thresholds for individual distortion
            dlist (list): Probability bounds given in eq. 5 of the paper for
                each threshold in clist
            verbose (bool): Verbosity flag
        """
        self.epsilon = epsilon
        self.dlist = dlist

        # main conditional map
        Pmap = Variable((self.dfP.shape[0], self.dfP.shape[1]))
        # marginal distribution of (Xh Yh)
        PXhYh = Variable((self.dfMask_Pxyd_to_Pxy.shape[1],))
        # rows represent p_(y|D)
        PYhgD = Variable((self.dfD_to_Y_address.shape[1],
                          self.dfD_to_Y_address.shape[0]))

        # marginal distribution
        dfMarginal = self.dfJoint.groupby(self.DXY_features)['Frequency'].sum()
        PxydMarginal = pd.concat([self.dfP, dfMarginal],
                                 axis=1).fillna(0)['Frequency'].values
        self.PxydMarginal = PxydMarginal
        PdMarginal = PxydMarginal.dot(self.dfMask_Pxyd_to_Pd).T
        PxyMarginal = PxydMarginal.dot(self.dfMask_Pxyd_to_Pxy).T

        # add constraints
        # 1. valid distribution
        constraints = [cp.sum(Pmap, axis=1) == 1]
        constraints.append(Pmap >= 0)

        # 2. definition of marginal PxhYh
        constraints.append(PXhYh == cp.sum(np.diag(PxydMarginal)*Pmap,
                                                axis=0).T)

        # add the conditional mapping
        constraints.append(
            PYhgD == np.diag(np.ravel(PdMarginal)**(-1)).dot(
                     self.dfMask_Pxyd_to_Pd.values.T).dot(
                     np.diag(PxydMarginal)) *
                     Pmap * self.dfMask_Pxy_to_Py.values)

        # 3. add excess distorion
        # print(PxyMarginal)
        # Pxy_xhyh = np.nan_to_num(np.diag(PxyMarginal**(-1))).dot(self.dfMask_Pxyd_to_Pxy.values.T).dot(np.diag(PxydMarginal))*Pmap
        Pxy_xhyh = np.nan_to_num(np.diag((PxyMarginal+1e-10)**(-1))).dot(
            self.dfMask_Pxyd_to_Pxy.values.T).dot(
            np.diag(PxydMarginal+1e-10)) * Pmap

        for i in range(len(self.CMlist)):
            constraints.append(
                cp.sum(cp.multiply(self.CMlist[i], Pxy_xhyh), axis=1) <=
                self.dlist[i])

    # 4. Discrimination control
        for d in range(self.dfMask_Pxyd_to_Pd.shape[1]):
            for d2 in range(self.dfMask_Pxyd_to_Pd.shape[1]):
                if d > d2:
                    continue
                # constraints.append(PYhgD[d,:].T<=PYhgD[d2,:].T*(1+self.epsilon))
                # constraints.append(PYhgD[d2,:].T<=PYhgD[d,:].T*(1+self.epsilon))
                constraints.append(
                    PYhgD[d, :].T - PYhgD[d2, :].T <= self.epsilon)
                constraints.append(
                    PYhgD[d2, :].T - PYhgD[d, :].T <= self.epsilon)

        # 5. Objective is l1 distance between the original
        # and perturbed distributions
        obj = Minimize(cp.norm(PXhYh-PxyMarginal, 1)/2)

        prob = Problem(obj, constraints)
        prob.solve(verbose=verbose)

        if prob.status in ["optimal", "optimal_inaccurate"]:
            print("Optimized Preprocessing: Objective converged to %f"
                % (prob.value))
        else:
            raise RuntimeError("Optimized Preprocessing: Optimization did not "
                               "converge")

        self.dfP.loc[:, :] = Pmap.value
        self.optimum = prob.value
        self.const = []

        for i in range(len(self.CMlist)):
            self.const.append(
                cp.sum(cp.multiply(self.CMlist[i], Pxy_xhyh),
                            axis=1).value.max())

    def compute_marginals(self):
        """Compute a bunch of required marginal distributions."""

        self.dfFull = pd.DataFrame(
            (np.diag(self.PxydMarginal)).dot(self.dfP.values),
            index=self.dfP.index, columns=self.dfP.columns)

        self.dfPyMarginal = pd.DataFrame(
            self.PxydMarginal.dot(self.dfMask_Pxyd_to_Py).T,
            index=self.dfMask_Pxyd_to_Py.columns)
        self.dfPdMarginal = pd.DataFrame(
            self.PxydMarginal.dot(self.dfMask_Pxyd_to_Pd).T,
            index=self.dfMask_Pxyd_to_Pd.columns)
        self.dfPxyMarginal = pd.DataFrame(
            self.PxydMarginal.dot(self.dfMask_Pxyd_to_Pxy).T,
            index=self.dfMask_Pxyd_to_Pxy.columns)

        self.dfPyhgD = pd.DataFrame(
            np.diag(np.ravel(self.dfPdMarginal.values)**(-1)).dot(
                self.dfMask_Pxyd_to_Pd.values.T).dot(
                self.dfFull.values).dot(
                self.dfMask_Pxy_to_Py.values),
            index=self.dfPdMarginal.index,
            columns=self.dfMask_Pxy_to_Py.columns)

        self.dfPxydMarginal = pd.DataFrame(self.PxydMarginal,
                                           index=self.dfMask_Pxyd_to_Pxy.index)

        self.dfPxygdPrior = self.dfPxydMarginal.reset_index().groupby(
            self.D_features+self.Y_features)[0].sum().unstack(self.Y_features)
        self.dfPxygdPrior = self.dfPxygdPrior.div(self.dfPxygdPrior.sum(axis=1),
                                                  axis=0)
