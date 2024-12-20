# Identifying Aspect-Regimes for Enhanced ESG-Investing Through News Data

This is the code repository accompanying the paper **Identifying Aspect-Regimes for Enhanced ESG-Investing Through News Data**.

In order to run the notebooks, we recommend creating a new environment with python=3.10 . Then, you can install the packages in the ``requirements.txt`` file using

```
pip install -r requirements.txt
```

Afterwards, you need to get the ESG data from the [Nano-ESG](https://github.com/Bailefan/Nano-ESG) repository. Download the file under ``data/full_data/nano_esg.json``.

Go through the notebook, ``dtw_esg_sentiment.ipynb``, in order to replicate Figure 1 and 2, or to look at other companies rather than the example of *daimler* we used in the paper.

Be aware that the paper derived its results from a different source of price data, which we are not allowed to share. In order to be able to replicate the results at all, we implemented a new version using ``yfinance`` in this notebook. For this reason, the results here might be slightly different from the ones presented in the paper, but should overall be very similar.

In addition, we can not share our portfolio optimization pipeline, which is private. Because of this, it is not possible to replicate Figures 3 and 4 from the paper, unless you have your own optimization pipeline.
We do share the estimator files for a wide combination of hyperparameters under the folder ``hp_opt``. In addition, a cell in the notebook can replicate the creation of the estimator files, if you wish to test with other parameters.