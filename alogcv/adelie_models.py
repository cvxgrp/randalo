from alogcv.models import ALOModel, LinearMixin, SeparableRegularizerMixin
import adelie as ad
import numpy as np
import torch



class AdelieLassoModel(LinearMixin, SeparableRegularizerMixin, ALOModel):
    """
    Lasso model using Adelie.
    """

    def __init__(self, lamda):
        super().__init__()
        self.lamda = lamda
        self._direct = False

    def get_new_model(self):
        class C:
            coef_ = None
            def fit(s, X, y):
                state = ad.solver.grpnet(
                        X=np.asfortranarray(X),
                        glm=ad.glm.gaussian(y),
                        lmda_path = np.array([self.lamda]),
                        intercept=False,
                )
                s.coef_ = state.betas.toarray()[0]
            def predict(s, x):
                return x @ s.coef_

        return C()


    @property
    def reg_hessian_diag_(self):
        self._fitted_check()
        hess = np.zeros(self.X.shape[1])
        hess[np.abs(self.model.coef_) <= 1e-8] = float("inf")
        return torch.tensor(hess)

    @staticmethod
    def loss_fun(y, y_hat):
        return (y - y_hat) ** 2 / 2
