"""
Script to twitch plotting so that c-star, the full vector of clusters, is also \
        included in the singlevar and dualvar plots
"""

import paper_model as pm

data_path = '../paper_run_2/'
models = ['LR', 'RF', 'ABC', 'QDA']
y_vars = ['ovrl', 'ov_mean']
rankeds = [True]

for y_var in y_vars:
    for ranked in rankeds:
        r = 'r' if ranked else 'nr'
        print('Plotting {}-{}'.format(y_var, r))
        save_path = data_path + y_var + '-r/' if ranked else data_path + y_var + '-nr/'
        PTS = pm.PredictTieStrength(y_var=y_var, data_path=data_path, save_prefix=save_path, \
                    alpha_step=5, ranked=ranked, models=models)
        PTS.load_scores(True, False, False)
        PTS.get_alphas()

        save_path += 'cluster/'
        PTS_c = pm.PredictTieStrength(y_var=y_var, data_path=data_path, save_prefix=save_path, \
                    alpha_step=5, ranked=ranked, models=models)
        PTS_c.load_scores(False, False, True)
        PTS.variables.append('c_star')
        for model, score in PTS_c.full_scores.items():
            PTS.single_scores[model]['c_star'] = score

        PTS.merge_scores()
        average = PTS.average_score.copy()
        PTS.plot_all_single()
        save_path = data_path + 'c_star/' + y_var + '-r/' if ranked else data_path + 'c_star/' + y_var + '-nr/'
        PTS = pm.PredictTieStrength(y_var=y_var, data_path=data_path, save_prefix=save_path, \
                    alpha_step=5, ranked=ranked, models=models, c_star=True)
        PTS.load_scores(False, True, False)
        PTS.get_alphas()
        PTS.average_score = average
        PTS.merge_scores()
        PTS.plot_dual_var()

