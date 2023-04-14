import numpy as np


class ModelNotTrainedError(Exception):
    def __init__(self):
        self.message = "Model has not been trained yet. Please fit the model before attempting to make predictions."
        super().__init__(self.message)


class StochasticGradientRegressor:
    """
    Regressor for linear regression using stochastic gradient descent.


    Methods
    -------
    fit(X, y)
        Fit the linear regression model to the given data.
    predict(X)
        Predict the targets for the input data X using the trained linear classifier.
    """

    def __init__(self, cost_func='mse', regularization=None, reg_lambda=0.01, learning_rate=0.1,
                 adaptive_lr=False, decay_rate=0.9, max_epochs=1000, batch_size=32, epsilon=1e-7,
                 random_state=None, verbose=True):
        """
        Initializes the regressor with the given hyperparameters.

        Parameters:
        -----------
        cost_func : str, optional (default='mse')
            The cost function to use. Can be one of 'mse' (mean squared error), 'rmse' (root mean
            squared error), 'mae' (mean absolute error), or 'logistic' (logistic regression).
        regularization : str or None, optional (default=None)
            The regularization method to use. Can be one of 'l1', 'l2', 'elastic_net', or None.
        reg_lambda : float, optional (default=0.01)
            The regularization strength (lambda) parameter.
        learning_rate : float, optional (default=0.01)
            The learning rate for the gradient descent update.
        adaptive_lr : bool, optional (default=False)
            Whether to use adaptive learning rate.
        decay_rate : float, optional (default=0.9)
            The decay rate for the adaptive learning rate update.
        max_epochs : int, optional (default=1000)
            The maximum number of epochs (iterations) to run the gradient descent algorithm.
        batch_size : int, optional (default=32)
            The batch size for mini-batch gradient descent.
        epsilon : float, optional (default=1e-7)
            A small value used to prevent division by zero.
        random_state : int or None, optional (default=None)
            The random seed to use for shuffling the data.
        verbose : bool, optional (default=True)
            Whether to print progress messages during training.
        """

        self.cost_func = cost_func
        self.regularization = regularization
        self.reg_lambda = reg_lambda
        self.learning_rate = learning_rate
        self.adaptive_lr = adaptive_lr
        self.decay_rate = decay_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.random_state = random_state
        self.verbose = verbose
        self.w, self.b = None, None

        if self.adaptive_lr:
            self.lr_schedule = learning_rate

    def _calculate_cost(self, X, y, w, b):
        """
        Calculate the cost function for a given set of inputs and weights.
        """
        n_samples = X.shape[0]
        y_pred = X.dot(w) + b
        cost = None

        if self.cost_func == 'mse':
            cost = 1 / (2 * n_samples) * np.sum((y_pred - y) ** 2)
        elif self.cost_func == 'rmse':
            cost = np.sqrt(1 / n_samples * np.sum((y_pred - y) ** 2))
        elif self.cost_func == 'mae':
            cost = 1 / n_samples * np.sum(np.abs(y_pred - y))
        elif self.cost_func == 'logistic':
            cost = 1 / n_samples * np.sum(np.log(1 + np.exp(-y * y_pred)))

        if self.regularization == 'l1':
            cost += self.reg_lambda * np.sum(np.abs(w))
        elif self.regularization == 'l2':
            cost += self.reg_lambda * np.sum(w ** 2)
        elif self.regularization == 'elastic_net':
            cost += self.reg_lambda * (1 - self.learning_rate) * np.sum(
                w ** 2) + self.reg_lambda * self.learning_rate * np.sum(np.abs(w))

        return cost

    def _calculate_gradient(self, X, y, w, b):
        """
        Calculate the gradient of the cost function for a given set of inputs, weights, and bias.
        """
        n_samples = X.shape[0]
        y_pred = X.dot(w) + b
        dJ_dw, dJ_db = None, None

        if self.cost_func == 'mse':
            dJ_dw = 1 / n_samples * (X.T.dot(y_pred - y))
            dJ_db = 1 / n_samples * np.sum(y_pred - y)
        elif self.cost_func == 'rmse':
            dJ_dw = 1 / n_samples * (X.T.dot(y_pred - y)) / np.sqrt(np.sum((y_pred - y) ** 2))
            dJ_db = 1 / n_samples * np.sum((y_pred - y)) / np.sqrt(np.sum((y_pred - y) ** 2))
        elif self.cost_func == 'mae':
            dJ_dw = 1 / n_samples * (X.T.dot(np.sign(y_pred - y)))
            dJ_db = 1 / n_samples * np.sum(np.sign(y_pred - y))
        elif self.cost_func == 'logistic':
            dJ_dw = 1 / n_samples * (X.T.dot(y_pred - y) * y) / (1 + np.exp(y * y_pred))
            dJ_db = 1 / n_samples * np.sum((y_pred - y) * y) / (1 + np.exp(y * y_pred))

        if self.regularization == 'l1':
            dJ_dw += self.reg_lambda * np.sign(w)
        elif self.regularization == 'l2':
            dJ_dw += 2 * self.reg_lambda * w
        elif self.regularization == 'elastic_net':
            dJ_dw += self.reg_lambda * ((1 - self.learning_rate) * 2 * w + self.learning_rate * np.sign(w))

        return dJ_dw, dJ_db

    def _update_learning_rate(self, epoch):
        """
        Updates the learning rate for the given epoch if adaptive learning rate is enabled.
        """
        if self.adaptive_lr:
            self.learning_rate = self.lr_schedule / (1 + self.decay_rate * epoch)

    def fit(self, X, y):
        """
        Fits the model to the given data.
        Parameters:
        -----------
        X : ndarray
            The feature matrix with shape (n_samples, n_features).
        y : ndarray
            The target variable with shape (n_samples,).

        Returns:
        --------
        w : ndarray
            The learned weight vector with shape (n_features,).
        b : float
            The learned bias term.
        costs : list
            The training cost at each epoch.

        """
        rng = np.random.RandomState(self.random_state)
        n_samples, n_features = X.shape
        w = np.zeros(n_features)
        b = 0
        costs = []

        for epoch in range(self.max_epochs):
            if self.adaptive_lr:
                self._update_learning_rate(epoch)

            # Shuffle the data
            permutation = rng.permutation(n_samples)
            X = X[permutation]
            y = y[permutation]

            for i in range(0, n_samples, self.batch_size):
                # Get the current batch
                X_batch = X[i:i + self.batch_size]
                y_batch = y[i:i + self.batch_size]

                # Calculate the gradient and cost
                dJ_dw, dJ_db = self._calculate_gradient(X_batch, y_batch, w, b)
                cost = self._calculate_cost(X_batch, y_batch, w, b)

                # Update the weights and bias
                w -= self.learning_rate * dJ_dw
                b -= self.learning_rate * dJ_db

            # Calculate the cost for the entire dataset and append to the costs list
            cost = self._calculate_cost(X, y, w, b)
            costs.append(cost)

            if self.verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}: cost = {cost}")

            self.w = w
            self.b = b
        return w, b, costs

    def predict(self, X):
        """
        Predict the targets for the input data X using the trained linear regressor.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        y_pred : array, shape (n_samples,)
            Predicted class labels for the input data X.

        Raises
        ------
        NotFittedError : if self.w or self.b are None
            If the linear regressor has not been fitted (i.e., trained) yet,
            attempting to predict with it will raise a NotFittedError.
        """

        if self.w is None or self.b is None:
            raise ModelNotTrainedError
        return X.dot(self.w) + self.b


class StochasticGradientSVM:
    """
    Support Vector Machine (SVM) for binary classification using stochastic gradient descent.

    Methods
    -------
    fit(X, y)
        Fit the classification model to the given data.
    predict(X)
        Predict the class labels for the input data X using the trained linear classifier.
    """

    def __init__(self, C=1.0, tol=1e-4, max_epochs=1000, kernel='linear', gamma='scale', coef0=0.0, degree=3,
                 learning_rate=0.0001, batch_size=100, regularization=None, reg_lambda=1.0,
                 random_state=None, verbose=False):
        """
        Initializes a StochasticGradientSVM object.

        Parameters
        ----------
        C : float, default=1.0
            Penalty parameter C of the error term.
        tol : float, default=1e-4
            Tolerance for stopping criterion.
        max_epochs : int, default=1000
            Maximum number of epochs (passes over the training data).
        kernel : str, default='linear'
            Type of kernel to use ('linear', 'rbf', 'poly' or 'sigmoid').
        gamma : {'scale', 'auto'} or float, default='scale'
            Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
        coef0 : float, default=0.0
            Independent term in kernel function. It is only significant in 'poly' and 'sigmoid'.
        degree : int, default=3
            Degree of the polynomial kernel function ('poly'). Ignored by all other kernels.
        learning_rate : float, default=0.0001
            Learning rate for the gradient descent optimizer.
        batch_size : int, default=100
            Number of samples to use in each batch for stochastic gradient descent.
        regularization : {'l1', 'l2', 'elastic_net', None}, default=None
            Type of regularization to apply on the weights.
        reg_lambda : float, default=1.0
            Regularization strength. Larger values imply stronger regularization.
        random_state : int or None, default=None
            Seed of the pseudo random number generator used for shuffling the data.
        verbose : bool, default=False
            Whether to print progress messages during training.
        """

        self.C = C
        self.tol = tol
        self.max_epochs = max_epochs
        self.kernel = kernel
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.regularization = regularization
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.verbose = verbose
        self.X_fit_ = None
        self.beta = None
        self.beta_0 = None

    def _compute_kernel(self, X1, X2=None):
        """
        Compute the kernel matrix between X1 and X2.

        Parameters
        ----------
        X1: array-like of shape (n_samples_1, n_features)
            First data matrix.
        X2: array-like of shape (n_samples_2, n_features), default=None
            Second data matrix. If None, use X1.

        Returns
        -------
        K: ndarray of shape (n_samples_1, n_samples_2)
            Kernel matrix.
        """

        if X2 is None:
            X2 = X1
        if self.kernel == 'linear':
            return np.dot(X1, X2.T)
        elif self.kernel == 'rbf':
            if self.gamma == 'scale':
                gamma = 1 / (X1.shape[1] * X1.var())
            elif self.gamma == 'auto':
                gamma = 1 / X1.shape[1]
            else:
                gamma = self.gamma
            XX1T = np.dot(X1, X2.T)
            XX2T = np.dot(X2, X2.T)
            XX1_sqnorms = np.diag(XX1T)
            XX2_sqnorms = np.diag(XX2T)
            K = -2 * XX1T + XX1_sqnorms[:, np.newaxis] + XX2_sqnorms[np.newaxis, :]
            K *= -gamma
            np.exp(K,K)
            return K
        elif self.kernel == 'poly':
            return (np.dot(X1, X2.T) + self.coef0)**self.degree
        elif self.kernel == 'sigmoid':
            return np.tanh(np.dot(X1, X2.T)*self.gamma+self.coef0)

    def _compute_loss(self, X, y):
        """
        Computes the hinge loss and regularization loss.

        Parameters
        ----------
        X : array-like of shape (n_samples,n_features)
            Training data.
        y: array-like of shape(n_samples,)
            Target values.

        Returns
        -------
        loss: float
            The hinge loss plus the regularization loss.
        """

        K = self._compute_kernel(X)
        margin = 1 - y * (np.dot(K, self.beta) + self.beta_0)
        hinge_loss = np.maximum(0, margin) ** 2
        if self.regularization == 'l1':
            reg_loss = self.reg_lambda * np.abs(self.beta).sum()
        elif self.regularization == 'l2':
            reg_loss = 0.5 * self.reg_lambda * (self.beta ** 2).sum()
        elif self.regularization == 'elastic_net':
            reg_loss = self.reg_lambda * (0.5 * (self.beta ** 2).sum() + np.abs(self.beta).sum())
        else:
            reg_loss = 0
        return self.C * hinge_loss.sum() + reg_loss

    def _compute_gradient(self, K, y, indices):
        """
        Computes the gradient of the loss function with respect to beta and beta_0.

        Parameters
        ----------
        K: ndarray of shape (n_samples, n_samples)
            Kernel matrix.
        y: array-like of shape (n_samples,)
            Labels.
        indices: array-like of shape (batch_size,)
            Indices of the current batch.

        Returns
        -------
        d_beta: ndarray of shape (n_samples,)
            Gradient with respect to beta.
        d_beta_0: float
            Gradient with respect to beta_0.
        """

        margin = 1 - y[indices] * (np.dot(K[indices], self.beta) + self.beta_0)
        margin = margin[:, np.newaxis]
        d_beta = -2 * self.C * np.mean((margin > 0) * y[indices][:, np.newaxis] * K[indices], axis=0)
        d_beta_0 = -2 * self.C * np.mean((margin > 0) * y[indices])
        if self.regularization == 'l1':
            d_beta += self.reg_lambda * np.sign(self.beta)
        elif self.regularization == 'l2':
            d_beta += self.reg_lambda * self.beta
        elif self.regularization == 'elastic_net':
            d_beta += self.reg_lambda * (self.beta + np.sign(self.beta))
        return d_beta, d_beta_0

    def fit(self, X, y):
        """
        Fit the SVM model according to the given training data.

        Parameters
        ----------
        X: array-like of shape(n_samples,n_features)
            Training data.
        y: array-like of shape(n_samples,)
            Target values.

        Returns
        -------
        beta: ndarray of shape (n_samples,)
            Returns beta vector
        d_beta_0: float
            Returns beta_0
        """

        n_samples, n_features = X.shape
        K = self._compute_kernel(X)
        self.X_fit_ = X
        self.beta = np.zeros(n_samples)
        self.beta_0 = 0
        rng = np.random.default_rng(self.random_state)
        for epoch in range(self.max_epochs):
            shuffled_indices = rng.permutation(n_samples)
            for i in range(0, n_samples, self.batch_size):
                indices = shuffled_indices[i:i + self.batch_size]
                d_beta, d_beta_0 = self._compute_gradient(K, y, indices)
                self.beta -= self.learning_rate*d_beta
                self.beta_0 -= self.learning_rate*d_beta_0

            if epoch % 100 == 0 and epoch > 0 and self.verbose:
                loss = self._compute_loss(K, y)
                print(f'Epoch {epoch}/{self.max_epochs} | Loss: {loss:.6f}')
        return self.beta, self.beta_0

    def predict(self, X):
        """
        Perform classification on samples in X.

        Parameters
        ----------
        X: array-like of shape(n_samples,n_features)
          Samples.

        Returns
        -------
        y_pred: array-like of shape(n_samples,)
            Class labels for samples in X.

        Raises
        ------
        ModelNotTrainedError : if self.beta or self.beta_0 are None
            If the linear classifier has not been fitted (i.e., trained) yet,
            attempting to predict with it will raise a ModelNotTrainedError.
        """

        if self.beta is None or self.beta_0 is None:
            raise ModelNotTrainedError
        K = self._compute_kernel(self.X_fit_, X)
        return np.sign(np.dot(K.T, self.beta) + self.beta_0)
