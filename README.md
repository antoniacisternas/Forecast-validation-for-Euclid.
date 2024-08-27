To estimate the uncertainties in cosmological parameters, we utilize the Fisher matrix formalism. Our analysis incorporates three cosmological probes:

- Spectroscopic Galaxy Clustering (GCspec)
- Photometric Galaxy Clustering (GCphoto)
- Weak Lensing (WL)
  
In addition to the individual contributions from each probe, we compute the cross-correlation between the photometric galaxy clustering (GCphoto) and weak lensing (WL) datasets. The final Fisher matrix is constructed by combining the Fisher matrices derived from the GCspec data with those from the GCphoto-WL cross-correlation. This comprehensive Fisher matrix provides a robust estimate of the constraints on the cosmological parameters, accounting for both individual and joint probe contributions

Documents:
For GCspec:
- 'Class_fisher_matrix_pessimistic.ipynb' has the Fisher matrix difining the functions with Object-oriented programming.
- 'Fisher_matrix_optimistic.ipynb' has the Fisher matrix for an optimistic setting for a w0, wa, gamma and non-flat model.
- 'Fisher_matrix_pessimistic.ipynb' has the Fisher matrix for a pessimistic setting for a w0, wa, gamma and non-flat model.
