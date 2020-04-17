"""
Script to twitch plotting so that c-star, the full vector of clusters, is also \
        included in the singlevar and dualvar plots
"""

import paper_model as pm

data_path = '../paper_run/'
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
        PTS.load_scores(True, False, False) #CHANGE this when we have dual scores
        PTS.get_alphas()

        save_path += 'cluster/'
        PTS_c = pm.PredictTieStrength(y_var=y_var, data_path=data_path, save_prefix=save_path, \
                    alpha_step=5, ranked=ranked, models=models)
        PTS_c.load_scores(False, False, True)
        PTS.variables.append('c_star')
        for model, score in PTS_c.full_scores.items():
            PTS.single_scores[model]['c_star'] = score

        PTS.merge_scores()
        PTS.plot_all_single()


        #PTS.plot_full_scores()
        for model in models:
            try:
                pass #PTS.plot_feature_imp(case=model)
            except:
                print('Could not plot FI for model {}'.format(model))

