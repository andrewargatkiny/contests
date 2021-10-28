# ML challenge Yandex Cup 2021 (RecSys)

## The task
See [here](problem_statement.md) (in Russian)

## Solution

I've tried 2 approaches:

1) Refined matrix factorization method with implicit WARP loss in LightFM library. 

[LightFM.ipynb](LightFM.ipynb)

2) Gradient boosting with with LambdaRank loss function and pretty heavy feature engineering.

[Gradient Boosting.ipynb"](gradient_boosting.ipynb")

I entered the contest too late so I didn't have enough time to test more hypotheses and build two-stage solution using lightFM users' and orgs' embeddings as inputs for LightGBM.

The trick that improves the score by a highest margin is to restrict set of the orgs that are put into the ranking model for final prediction by N most popular orgs among tourists in last 500 days.

Keep in mind that it's not production-grade code which I didn't intend to upload till now so it can be messy at some places. It also contains remnants of baseline solution provided by Yandex.
