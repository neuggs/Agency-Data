# Agency Data
Predictive model for insurance agency data.

## Introduction
Agency data intuitively has value to an agency and potentially to agent's carrier partner(s). One way to gain value from that data - which exists within agency management systems - is to create one or more predictive models. Gaining insight from predictive modeling using agency data is the fundamental nature of this effort.

## Requirements
This entire project was completed using RStudio [RStudio](https://www.rstudio.com/) and Python (Jupyter Notebook). You'll need R, an R IDE (if you prefer), and a Python distribution that includes Jupyter Notebook. I used RStudio and Anaconda (with PyCharm).

You'll need the following R packages installed using `install packages("PACKAGE_NAME")`:
* xlsx
* dplyr
* e1071
* caret
* formattable

There should be no issue installing these packages if you have RStudio.

You'll need the following Python libraries installed using (`pip install <PACKAGE_NAME>`):
* pandas
* numpy
* sklearn

## Epilogue
It's clear that predictive modelling with agency has value. The contents of this analysis are a single perspective of such an effort, although subsequently, at least one other perspective is presented to give the reader a glimpse into how varied this analysis could potentially be.

This uses a largely unalered simple Naive Bayes model to set a baseline for subsequent predictive models, which can be created using other algorithms.
