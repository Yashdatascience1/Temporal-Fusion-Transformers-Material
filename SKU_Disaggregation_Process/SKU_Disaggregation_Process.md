# SKU Disaggregation Process Document
## How Family-Level Predictions Are Split to SKU Level

---

## The Problem

We have sales predictions at the **dealer + model family** level (e.g., "Dealer 10878, SPLENDOR+ DRUM SELF CAST RED BLACK"). But business needs the prediction at the **individual SKU level** — each specific product variant within that family.

---

## Step-by-Step Process

### Step 1: Identify Active SKUs in Each Family

Each model family contains multiple SKUs (product variants). We look up the supercedence mapping table to find which SKUs are currently **active** (not discontinued) within each family. Discontinued SKUs are ignored entirely.

For example, the family "SPLENDOR+ DRUM SELF CAST RED BLACK" might have 3 active SKUs.

---

### Step 2: Get Historical Sales Proportions

We pull **3 months of actual retail sales** from the Customer Retails table (Individual customers only), ending just before the earliest prediction month. This tells us how each SKU within a family has been selling recently at each dealer.

For example, if Dealer 10878 sold:
- SKU-A: 60 units
- SKU-B: 30 units
- SKU-C: 10 units
- **Total: 100 units**

Then the proportions are: SKU-A = 60%, SKU-B = 30%, SKU-C = 10%.

---

### Step 3: Handle OBD (On-Board Diagnostics) Remapping

Some SKUs have been superseded by newer variants (OBD2 compliance changes). When this happens, we map the old SKU's sales and stock to the current replacement SKU, so we're always working with the latest product codes.

---

### Step 4: Map Child Dealers to Parent Dealers

Predictions are at the **parent dealer** level (a dealer group), but sales data is at the **child dealer** level (individual showrooms). We roll up child dealer sales to the parent dealer level before calculating proportions.

---

### Step 5: Filter Out Small Dealers

Any dealer group with fewer than **50 total units sold** in the 3-month window is removed. These low-volume dealers don't have enough sales history to produce reliable proportions.

---

### Step 6: Split the Family Prediction to SKUs

This is the core step. Three scenarios:

| Scenario | What Happens |
|---|---|
| Family has **1 active SKU** | The entire family prediction goes to that single SKU |
| Family has **multiple active SKUs but no recent sales** at this dealer | Prediction is split **equally** among all active SKUs (1/N each) |
| Family has **multiple active SKUs with sales history** | Prediction is split **proportionally** based on each SKU's share of the dealer's recent sales |

A SKU with zero sales at that dealer (but the family had sales overall) gets **zero** predicted sales — the assumption is that dealer doesn't carry that variant.

---

### Step 7: Aggregate Daily Predictions to Monthly

The input predictions are daily. Since actual sales comparisons happen at the monthly level, we sum up daily predictions into monthly totals before comparing.

---

### Step 8: Fetch Actual Sales for Comparison

We pull actual sales from the Customer Retails table for the period between the earliest prediction date and the latest date where actual data is available. This gives us the ground truth to measure accuracy. The same OBD remapping and parent-dealer rollup are applied.

---

### Step 9: Remove Zero-Actual Records

Any SKU-dealer-month combination where actual sales were zero is dropped. We can't measure prediction accuracy when there were no real sales.

---

### Step 10: Calculate Accuracy (MAPE)

For each remaining row, we compute:

> **MAPE** = |Predicted - Actual| / Actual x 100

This tells us the percentage error. Lower is better.

---

### Step 11: Classify SKUs into A / B / C Categories

Using standard **Pareto (ABC) analysis** on total actual sales across all dealers:

| Category | Rule | Meaning |
|---|---|---|
| **A** | Top 80% of cumulative sales | High-volume SKUs — the vital few |
| **B** | Next 15% (80-95% cumulative) | Medium-volume SKUs |
| **C** | Bottom 5% (95-100% cumulative) | Low-volume SKUs — the trivial many |

SKUs are ranked from highest to lowest total sales, and the cumulative percentage determines the cutoff.

---

## What the Output Table Contains

Each row represents one **dealer x SKU x month** combination, with:

- The original family prediction and how much was allocated to this SKU
- The actual sales for comparison
- The accuracy (MAPE %)
- The ABC category of that SKU
- The timeframes used for both the proportion calculation and the actuals comparison
- The prediction quantile and iteration details used as input filters

---

## Key Tables Used

| Table | Purpose |
|---|---|
| Input prediction table (parameter) | Family-level daily predictions |
| ANALYTICS_DATABASE.ANALYTICS_SALES.CUSTOMER_RETAILS | Actual retail sales (for proportions and accuracy) |
| MOP_DATABASE.SOQ.SKU_SUPERCEDENCE_MODEL_FAMILY_JUN_2026_UPDATED_V2 | SKU-to-family mapping and active/inactive status |
| MOP_DATABASE.SOQ.OBD2_MAPPING_VIEW | Old SKU to new SKU remapping (OBD2 compliance) |
| FIVETRAN_DATABASE.ORACLE_LDP_OLAP_SCHEMA.WC_INT_ORG_DH | Child dealer to parent dealer mapping |

---

## Key Filters Applied

1. Only **Individual** customer type sales are considered
2. Dealers with **fewer than 50 units** sold in the 3-month ECR window are excluded
3. Records where **actual sales = 0** are removed from the output
4. Only **active** SKUs (not discontinued) are included in the disaggregation
