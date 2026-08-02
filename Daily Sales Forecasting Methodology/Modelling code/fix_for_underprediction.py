Replacing this - 

lo = float(tr_sales.min())

            hi = float(tr_sales.max())

            if hi - lo < 1e-8:

                hi = lo + 1.0            # constant series guard

				

with 

# divide by MEAN, not max. Min-max divides every ordinary day by the

            # festive peak, collapsing typical values to ~0.02 where Huber's

            # gradient (= the error) is too small to learn from.

            lo = 0.0

            hi = float(tr_sales.mean())

            if hi < 1e-3:

                hi = 1.0                 # all-zero / near-dead series guard

is done.

---------------------------------------------

Now I will create a new cell right after the cell which has scaler_stats = manifest["scaler_stats"]

AND enter this. 

import numpy as np, os, json

new_stats = {}

for key in manifest["series_keys"]:

    with np.load(os.path.join(CACHE_DIR, f"{safe_name(key)}.npz")) as z:

        s = z["train_sales"]

    m = float(s.mean())

    new_stats[key] = [0.0, m if m > 1e-3 else 1.0]

manifest["scaler_stats"] = new_stats

with open(os.path.join(CACHE_DIR, "manifest.json"), "w") as f:

    json.dump(manifest, f)

scaler_stats = new_stats

print("scaler_stats recomputed on mean")

----------------------------------------

In the model configuration, I will use - use_reversible_instance_norm as True

------------------------------------------

Change the loss function to 

class HuberMaeFeatureLoss(nn.HuberLoss):

    """Huber on component 0, with two festive adjustments driven by component 1

    of the target (the 0/1 festive flag):

      1. festive days are weighted `festive_weight` times more heavily

      2. UNDER-prediction on festive days carries an extra one-sided penalty

    The one-sidedness is the point. The previous version added a symmetric

    `flag * |y - y_hat|`, which penalised over- and under-shoot equally and so

    pulled festive predictions toward the conditional MEDIAN -- downward on a

    right-skewed festive distribution, i.e. the opposite of the intent.

    """

    def __init__(self, delta=1.0, festive_weight=3.0, under_weight=3.0,

                 reduction='mean'):

        super().__init__(reduction='none', delta=delta)

        self.festive_weight = festive_weight

        self.under_weight   = under_weight

        self.user_reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        y_hat = input[..., 0]

        y     = target[..., 0]

        flag  = target[..., 1]
		
		if not hasattr(self, "_dbg"):
            print("flag uniques:", torch.unique(flag)[:8])
            print(f"y range: min {float(y.min()):.2f} max {float(y.max()):.2f} mean {float(y.mean()):.2f}")
            self._dbg = True

        base       = super().forward(y_hat, y)

        weighted   = (1.0 + self.festive_weight * flag) * base

        undershoot = torch.clamp(y - y_hat, min=0)          # >0 only when too low

        total      = weighted + self.under_weight * flag * undershoot

        if self.user_reduction == 'mean':

            return total.mean()

        if self.user_reduction == 'sum':

            return total.sum()

        return total

		

-------

num_attention_heads has been changed to 4

--------

loss_fn=HuberMaeFeatureLoss(delta=5.0, festive_weight=3.0, under_weight=3.0),

-----------------------------

Setting OUTPUT_CHUNK_LENGTH as 154 

and TEST_HORIZON also as 154

----



patience has been reduced to 5 



----



also, help me what is the purpose of this - 


flag uniques: tensor([0., 1.], device='cuda:0')
y range: min 0.00 max 281.50 mean 0.59


import numpy as np, os
dead_all, dead_recent = [], []
for k in series_keys:
    with np.load(os.path.join(CACHE_DIR, f"{safe_name(k)}.npz")) as z:
        s = z["train_sales"]
    if s.sum() == 0:        dead_all.append(k)
    elif s[-180:].sum() == 0: dead_recent.append(k)
print(f"never sold in training : {len(dead_all):,} ({len(dead_all)/len(series_keys)*100:.1f}%)")
print(f"nothing in last 180d   : {len(dead_recent):,} ({len(dead_recent)/len(series_keys)*100:.1f}%)")