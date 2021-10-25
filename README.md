### Finding Regions of Heterogeneity in Decision-Making via Expected Conditional Covariance

Justin Lim, Christina X Ji, Michael Oberst, Saul Blecker, Leora Horwitz, and David Sontag. 2021. Finding Regions of Heterogeneity in Decision-Making via Expected Conditional Covariance. In Thirty-fifth Conference on Neural Information Processing Systems.

Individuals often make different decisions when faced with the same context, due to personal preferences and background.  For instance, judges may vary in their leniency towards certain drug-related offenses, and doctors may vary in their preference for how to start treatment for certain types of patients.  With these examples in mind, we present an algorithm for identifying types of contexts (e.g., types of cases or patients) with high inter-decision-maker disagreement.  We formalize this as a causal inference problem, seeking a region where the assignment of decision-maker has a large causal effect on the decision.  We give an iterative algorithm to find a region maximizing this objective and give a generalization bound for its performance. In a semi-synthetic experiment, we show that our algorithm recovers the correct region of disagreement accurately compared to baselines. Finally, we apply our algorithm to real-world healthcare datasets, recovering variation that aligns with existing clinical knowledge.

To run our algorithm, see `run_semisynth_exp_recover_beta.ipynb` for how to call `IterativeRegionEstimator.py`. The baselines and  our model are also implemented in `baselines.py`. Helper functions (e.g. for evaluation) are in `helpers.py`.

Please refer to the following steps to reproduce the experiments and figures in this paper:

0. To set-up the required packages, run `create_env.sh`, passing in a conda environment name. Then run `source activate` with the environment name to enter it.

1. To run the semi-synthetic experiment,
    1. Download the criminal justice dataset from https://github.com/stanford-policylab/recidivism-predictions
    2. Process the data using  `data_processing/semisynth_process_data.ipynb`.
    2. To run the iterative algorithm and baselines, run `python3 run_baselines_on_semisynth.py` with the product of the following arguments:
        1. type of model: Iterative, Direct, TarNet, ULearner, CausalForest
        2. number of agents: 2, 5, 10, 20, 40, 87 in our experiments
        3. subset: drug_possession, misdemeanor_under35
    3. Figures 1, 3, and 4 compare metrics for the methods. They can be produced by running `plot_semisynth.ipynb`.
    4. Figure 2 examines tuning the region size. `run_semisynth_exp_recoverbeta.ipynb` is a stand-alone notebook for reproducing it.
    5. Figures 5 and 6 examine convergence of the iterative algorithm. They can be produced by running `plot_convergence.ipynb`.
    6. Figures 7 and 8 examine how robust the iterative algorithm and direct baselines are to violations of the assumption that there are two agent groups. First, run `python3 run_robustness_semisynth_experiment.py` with the product of the following arguments:
        1. type of model: Iterative, Direct
        2. number of groups: 2, 3, 5, 10
        3. subset: drug_possession, misdemeanor_under35
       Note that the number of agents is fixed at 40. The figures can then be produced by running `plot_robustness.ipynb`.
    7. Note: Helper code that is called to generate semi-synthetic data is located in `semisynth_subsets.py`, `semisynth_dataloader.py`, and `semisynth_dataloader_robust.py`.
    
2. The real-world diabetes experiment uses proprietary data extracted using `generate_t2dm_cohort.sql` and `first_line.sql`.
    1. Select an outcome model from logistic regressions, decision trees, and random forests  based on AUC, calibration, and partial dependence plots. Figure 9 and the statistics in Table 2 that guided our selection of a random forest outcome model are produced in `select_outcome_model_for_diabetes_experiment.ipynb`.
    2. The experiment is run with `python3 run_baseline_models.py diabetes Iterative DecisionTree RandomForest`. Figure 10b,  the information needed to create Figures 10a, the statistics in Tables 1 and 3, and the fold consistency evaluation  will be outputted.
    3. Note: Data loading helper functions, including how data is split, are located in `real_data_loader.py`. Most of the functions called to generate the output are located in `realdata_analysis.py`.
    
3. The real-world Parkinson's experiment was run using open-access data.
    1. Download the data from https://www.ppmi-info.org/. 
    2. Run `python3 ppmi_feature_extraction.py` passing in the directory containing the downloaded raw data and directory where processed data will be outputted.
    3. Manually process the treatment data to correct for typos in the drug name and treatment date
    4. Run `process_parkinsons_data.ipynb` to gather the data for the experiment.
    5. The experiment is run with `python3 run_baseline_models.py ppmi Iterative DecisionTree`. The information for creating Figure 11 and Table 4 are outputted.