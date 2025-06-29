"""Test data parsing from Google Sheets"""

import sys
import os
import pandas as pd
import requests
from io import StringIO

# Add parent directory to path to import main module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from west_reservoir_tracker import SHEET_URL, REQUEST_TIMEOUT


def test_data_parsing() -> list:
    """Test function to identify non-parseable records from Google Sheets.

    Returns:
        list: List of problematic records, empty if all data is valid
    """
    print("🔍 Testing data parsing from Google Sheets...")

    try:
        # Get raw data from Google Sheets
        response = requests.get(SHEET_URL, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()

        df = pd.read_csv(StringIO(response.text))
        print(f"📊 Raw data shape: {df.shape}")
        print(f"📊 Raw columns: {list(df.columns)}")
        print(f"📊 First 3 rows of raw data:")
        print(df.head(3))

        # Standardize column names
        df.columns = ["Date", "Temperature"] + list(df.columns[2:])
        df = df[["Date", "Temperature"]]

        # Store original for comparison
        original_df = df.copy()

        # Try to parse dates and temperatures
        df["Date_Parsed"] = pd.to_datetime(
            df["Date"], format="%d/%m/%Y", errors="coerce"
        )
        df["Temperature_Parsed"] = pd.to_numeric(df["Temperature"], errors="coerce")

        # Find problematic records
        problematic_records = []

        # Check for date parsing failures
        date_failures = df[df["Date_Parsed"].isna()]
        if not date_failures.empty:
            print(f"\n❌ Found {len(date_failures)} rows with unparseable dates:")
            for idx, row in date_failures.iterrows():
                record = {
                    "row": idx,
                    "issue": "date_parsing",
                    "original_date": row["Date"],
                    "original_temp": row["Temperature"],
                }
                problematic_records.append(record)
                print(
                    f"   Row {idx}: Date='{row['Date']}', Temp='{row['Temperature']}'"
                )

        # Check for temperature parsing failures
        temp_failures = df[df["Temperature_Parsed"].isna()]
        if not temp_failures.empty:
            print(
                f"\n❌ Found {len(temp_failures)} rows with unparseable temperatures:"
            )
            for idx, row in temp_failures.iterrows():
                record = {
                    "row": idx,
                    "issue": "temperature_parsing",
                    "original_date": row["Date"],
                    "original_temp": row["Temperature"],
                }
                problematic_records.append(record)
                print(
                    f"   Row {idx}: Date='{row['Date']}', Temp='{row['Temperature']}'"
                )

        # Check for completely empty rows
        empty_rows = df[
            (df["Date"].isna() | (df["Date"] == ""))
            & (df["Temperature"].isna() | (df["Temperature"] == ""))
        ]
        if not empty_rows.empty:
            print(f"\n⚠️  Found {len(empty_rows)} completely empty rows:")
            for idx in empty_rows.index:
                record = {
                    "row": idx,
                    "issue": "empty_row",
                    "original_date": "",
                    "original_temp": "",
                }
                problematic_records.append(record)
                print(f"   Row {idx}: Empty row")

        # Summary
        total_issues = len(problematic_records)
        if total_issues == 0:
            print("\n✅ All data parsed successfully!")
            assert total_issues == 0, "No parsing issues found"
        else:
            print(f"\n❌ Found {total_issues} total problematic records")
            print("Assertion will fail - returning problematic records list")

        return problematic_records

    except Exception as e:
        print(f"❌ Error testing data: {e}")
        return [{"error": str(e)}]


if __name__ == "__main__":
    issues = test_data_parsing()
    if issues:
        print(f"\n📝 Returning {len(issues)} problematic records:")
        for issue in issues:
            print(f"   {issue}")
        exit(1)  # Exit with error code
    else:
        print("\n✅ All tests passed!")
        exit(0)
