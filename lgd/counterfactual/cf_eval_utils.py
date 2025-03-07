

import torch
#TODO CF Accuracy

def cf_eval(_true, _pred, logger):
    print('CF_ACC')

    true = torch.cat(self._true).squeeze(-1)
    pred_score = torch.cat(self._pred)

    #print('pred_score', pred_score)
    pred_int = self._get_pred_int(pred_score)
    print('pred_int', pred_int )
    if true.shape[0] < 1e7:  # AUROC computation for very large datasets is too slow.
        # TorchMetrics AUROC on GPU if available.
        auroc_score = auroc(pred_score.to(torch.device(cfg.accelerator)),
                            true.to(torch.device(cfg.accelerator)),
                            task='binary')
        if self.test_scores:
            # SK-learn version.
            try:
                r_a_score = roc_auc_score(true.cpu().numpy(),
                                            pred_score.cpu().numpy())
            except ValueError:
                r_a_score = 0.0
            assert np.isclose(float(auroc_score), r_a_score)
    else:
        auroc_score = 0.
    
    reformat = lambda x: round(float(x), cfg.round)
    res = {
        'cf_accuracy': reformat(accuracy_score(true, pred_int)),
    }
    return res
    


#TODO Fidelity

#TODO Modification ratio/Proximity

#TODO Size

#TODO 