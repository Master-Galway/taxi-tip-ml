# Data Directory

This folder contains **sample datasets** used for illustration and testing.  
The **full original NYC taxi dataset is not included** in this repository due to size constraints and licensing considerations.

## Files Included

- `2017_taxi.csv`  
  A small random sample of rows from the full 2017 NYC Yellow Taxi dataset.  
  This allows users to run the notebook or scripts in a lightweight environment.

- `predicted_means.csv`  
  A sample of the `predicted_means` dataset provided as part of the Google Advanced Data Analytics Professional Certificate training materials, which can be calculated mannually.
These files are intentionally small to make the repository portable and easy to clone.

---

## Full Dataset

You can download the complete dataset directly from the official source:

NYC Taxi & Limousine Commission – Trip Record Data  
https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

The original data includes millions of rows and should be stored outside the repository when running full-scale training.

---

## Notes

- The notebook and Python scripts work with either the sample data **or** the full dataset.
- When running locally with the full dataset, update file paths accordingly (see instructions in the main `README.md`).