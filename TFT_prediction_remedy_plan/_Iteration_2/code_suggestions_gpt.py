import darts
import inspect

print("Darts version:", darts.__version__)
print(inspect.getsource(type(model.likelihood)._distr_from_params))
print(inspect.getsource(type(model.likelihood).predict_likelihood_parameters))


Darts version: 0.40.0
    def _distr_from_params(self, params):
        mu, alpha = params
        r, p = NegativeBinomialLikelihood._get_r_and_p_from_mu_and_alpha(mu, alpha)
        return _NegativeBinomial(r, p)

    def predict_likelihood_parameters(self, model_output: torch.Tensor) -> torch.Tensor:
        """Overwrite the parent since the parameters are extracted in two steps."""
        mu, alpha = self._params_from_output(model_output)
        r, p = NegativeBinomialLikelihood._get_r_and_p_from_mu_and_alpha(mu, alpha)
        return torch.cat([r, p], dim=-1)