import polars as pl
import numpy as np
import pandas as pd
import lightgbm as lgb
import pyarrow
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# 转换数据类型，仍需修改
def set_table_dtypes(df: pl.DataFrame) -> pl.DataFrame:
    # implement here all desired dtypes for tables
    # the following is just an example
    for col in df.columns:
        # last letter of column name will help you determine the type
        if col[-1] in ("P", "A"):
            df = df.with_columns(pl.col(col).cast(pl.Float64).alias(col))

    return df

# 处理字符串等，需修改
def convert_strings(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:  
        if df[col].dtype.name in ['object', 'string']:
            df[col] = df[col].astype("string").astype('category')
            current_categories = df[col].cat.categories
            new_categories = current_categories.to_list() + ["Unknown"]
            new_dtype = pd.CategoricalDtype(categories=new_categories, ordered=True)
            df[col] = df[col].astype(new_dtype)
    return df

# 生成data
def get_data():
    train_basetable = pl.read_csv("train/train_base.csv")
    train_static = pl.concat(
        [
            pl.read_csv("train/train_static_0_0.csv").pipe(set_table_dtypes),
            pl.read_csv("train/train_static_0_1.csv").pipe(set_table_dtypes),
        ],
        how="vertical_relaxed",
    )
    train_static_cb = pl.read_csv("train/train_static_cb_0.csv").pipe(set_table_dtypes)
    train_person_1 = pl.read_csv("train/train_person_1.csv").pipe(set_table_dtypes) 
    train_credit_bureau_b_2 = pl.read_csv("train/train_credit_bureau_b_2.csv").pipe(set_table_dtypes)
    
    # We need to use aggregation functions in tables with depth > 1, so tables that contain num_group1 column or 
    # also num_group2 column.
    train_person_1_feats_1 = train_person_1.group_by("case_id").agg(
        pl.col("mainoccupationinc_384A").max().alias("mainoccupationinc_384A_max"),
        (pl.col("incometype_1044T") == "SELFEMPLOYED").max().alias("mainoccupationinc_384A_any_selfemployed")
    )

    # Here num_group1=0 has special meaning, it is the person who applied for the loan.
    train_person_1_feats_2 = train_person_1.select(["case_id", "num_group1", "housetype_905L"]).filter(
        pl.col("num_group1") == 0
    ).drop("num_group1").rename({"housetype_905L": "person_housetype"})

    # Here we have num_goup1 and num_group2, so we need to aggregate again.
    train_credit_bureau_b_2_feats = train_credit_bureau_b_2.group_by("case_id").agg(
        pl.col("pmts_pmtsoverdue_635A").max().alias("pmts_pmtsoverdue_635A_max"),
        (pl.col("pmts_dpdvalue_108P") > 31).max().alias("pmts_dpdvalue_108P_over31")
    )

    # We will process in this examples only A-type and M-type columns, so we need to select them.
    selected_static_cols = []
    for col in train_static.columns:
        if col[-1] in ("A", "M"):
            selected_static_cols.append(col)
    print(selected_static_cols)

    selected_static_cb_cols = []
    for col in train_static_cb.columns:
        if col[-1] in ("A", "M"):
            selected_static_cb_cols.append(col)
    print(selected_static_cb_cols)

    # Join all tables together.
    data = train_basetable.join(
        train_static.select(["case_id"]+selected_static_cols), how="left", on="case_id"
    ).join(
        train_static_cb.select(["case_id"]+selected_static_cb_cols), how="left", on="case_id"
    ).join(
        train_person_1_feats_1, how="left", on="case_id"
    ).join(
        train_person_1_feats_2, how="left", on="case_id"
    ).join(
        train_credit_bureau_b_2_feats, how="left", on="case_id"
    )
    
    test_basetable = pl.read_csv("test/test_base.csv")
    test_static = pl.concat(
        [
            pl.read_csv("test/test_static_0_0.csv").pipe(set_table_dtypes),
            pl.read_csv("test/test_static_0_1.csv").pipe(set_table_dtypes),
            pl.read_csv("test/test_static_0_2.csv").pipe(set_table_dtypes),
        ],
        how="vertical_relaxed",
    )
    test_static_cb = pl.read_csv("test/test_static_cb_0.csv").pipe(set_table_dtypes)
    test_person_1 = pl.read_csv("test/test_person_1.csv").pipe(set_table_dtypes) 
    test_credit_bureau_b_2 = pl.read_csv("test/test_credit_bureau_b_2.csv").pipe(set_table_dtypes)
    
    test_person_1_feats_1 = test_person_1.group_by("case_id").agg(
        pl.col("mainoccupationinc_384A").max().alias("mainoccupationinc_384A_max"),
        (pl.col("incometype_1044T") == "SELFEMPLOYED").max().alias("mainoccupationinc_384A_any_selfemployed")
    )

    test_person_1_feats_2 = test_person_1.select(["case_id", "num_group1", "housetype_905L"]).filter(
        pl.col("num_group1") == 0
    ).drop("num_group1").rename({"housetype_905L": "person_housetype"})

    test_credit_bureau_b_2_feats = test_credit_bureau_b_2.group_by("case_id").agg(
        pl.col("pmts_pmtsoverdue_635A").max().alias("pmts_pmtsoverdue_635A_max"),
        (pl.col("pmts_dpdvalue_108P") > 31).max().alias("pmts_dpdvalue_108P_over31")
    )

    data_submission = test_basetable.join(
        test_static.select(["case_id"]+selected_static_cols), how="left", on="case_id"
    ).join(
        test_static_cb.select(["case_id"]+selected_static_cb_cols), how="left", on="case_id"
    ).join(
        test_person_1_feats_1, how="left", on="case_id"
    ).join(
        test_person_1_feats_2, how="left", on="case_id"
    ).join(
        test_credit_bureau_b_2_feats, how="left", on="case_id"
    )
    
    return data, data_submission

# 拿到train test valid等
def get_splits(data):
    case_ids = data["case_id"].unique().shuffle(seed=1)
    case_ids_train, case_ids_test = train_test_split(case_ids, train_size=0.6, random_state=1)
    case_ids_valid, case_ids_test = train_test_split(case_ids_test, train_size=0.5, random_state=1)

    cols_pred = []
    for col in data.columns:
        if col[-1].isupper() and col[:-1].islower():
            cols_pred.append(col)

    print(cols_pred)

    def from_polars_to_pandas(case_ids: pl.DataFrame) -> pl.DataFrame:
        return (
            data.filter(pl.col("case_id").is_in(case_ids))[["case_id", "WEEK_NUM", "target"]].to_pandas(),
            data.filter(pl.col("case_id").is_in(case_ids))[cols_pred].to_pandas(),
            data.filter(pl.col("case_id").is_in(case_ids))["target"].to_pandas()
        )

    base_train, X_train, y_train = from_polars_to_pandas(case_ids_train)
    base_valid, X_valid, y_valid = from_polars_to_pandas(case_ids_valid)
    base_test, X_test, y_test = from_polars_to_pandas(case_ids_test)

    for df in [X_train, X_valid, X_test]:
        df = convert_strings(df)
        
    return base_train, X_train, y_train, base_valid, X_valid, y_valid, base_test, X_test, y_test, cols_pred