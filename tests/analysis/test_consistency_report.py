import numpy as np
import pandas as pd
import pytest

from trajectory_prediction.analysis.consistency_report import paired_intervals
from trajectory_prediction.online.consistency_artifacts import STRATEGIES
from trajectory_prediction.online.consistency_evaluation import CONDITIONS, SEEDS


def test_fixed_three_seed_scene_pairing_all_contrasts_and_incomplete_rejection():
    rows=[]
    for s,m in zip(SEEDS,[1,2,3],strict=True):
        for c in CONDITIONS:
            for n in STRATEGIES:
                for scene,count in [("small",11),("large",33)]:
                    d=-.1*m if n=="revision_adaptive" else 0.
                    rows.append({"seed":s,"condition":c,"strategy":n,"scene_name":scene,"update_count":count,
                        "ade_sum":(1+d)*count,"fde_sum":(2+d)*count,"jitter_sum":(1+d)*count})
    frame=pd.DataFrame(rows)
    result=paired_intervals(frame,{"balanced":"revision_adaptive"})
    assert len(result)==252
    row=next(x for x in result if x["condition"]=="overall" and x["contrast"]=="revision_adaptive-RH" and x["metric"]=="ade")
    assert row["delta"]==pytest.approx(-.2)
    np.testing.assert_allclose(row["ci95"],[-.2,-.2])
    with pytest.raises(ValueError,match="paired"):
        paired_intervals(frame.iloc[1:],{"balanced":"revision_adaptive"})
