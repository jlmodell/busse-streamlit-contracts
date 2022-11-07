import streamlit as st
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
import pandas as pd
from dotenv import load_dotenv
import os
from datetime import datetime
from pymongo import MongoClient

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
assert MONGODB_URI is not None, "MONGODB_URI is not set"

client = MongoClient(MONGODB_URI)
db = client.bussepricing

contracts = db.get_collection("contract_prices")
# costs = db.get_collection("costs")
# customers = db.get_collection("customers")

filter_contracts_by_date = [
    datetime(2022, 12, 1, 0, 0, 0),
    datetime(2050, 12, 31, 0, 0, 0),
]

df = pd.DataFrame(
    list(
        contracts.find(
            {
                "contractend": {
                    "$gt": filter_contracts_by_date[0],
                    "$lte": filter_contracts_by_date[1],
                }
            },
            {
                "_id": 0,
                "pendingchanges": 0,
                "customers": 0,
                "endusers": 0,
                "customersdetails": 0,
                "customerdetails": 0,
                "items": 0,
                "tags": 0,
                "administrationfee": 0,
                "tradefee": 0,
                "freightfudgepercase": 0,
                "overheadfudgepercase": 0,
                "commissionallocation": 0,
                "labor_safety_margin": 0,
                "material_safety_margin": 0,
                "overhead_safety_margin": 0,
            },
        ).sort("contractend", 1)
    )
)

df["pricingagreements"] = df["pricingagreements"].apply(
    lambda x: ",\n".join([f"{item['item']}: {item['price']:.2f}" for item in x])
)

df["URL"] = df["contractnumber"].apply(
    lambda x: f"http://busseweb.com:7000/web/v1/contract/{x}"
)


def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = st.checkbox("Add filters")

    if not modify:
        return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].astype(str).str.contains(user_text_input)]

    return df


df = filter_dataframe(df)

chart_data = df.groupby("contractend").count()["contractnumber"]

PASSWORD = os.getenv("ACCESS_PASS")
assert PASSWORD is not None, "ACCESS_PASS is not set"


def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == PASSWORD:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True


if check_password():
    st.bar_chart(chart_data)
    st.dataframe(df)
