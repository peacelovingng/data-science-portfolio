# Poverty in The United States, 2017

This project provided storytelling and visualizations regarding poverty in The United States for year 2017 using census data. The purpose of this project is to use visualizations to convey a message to a broader range of audiences. 

### Project Motivation
For this project, I was interestested in analyzing census data regarding poverty in to better understand:

1. What percentage of the population is affected by poverty?

2. What states are most affected by poverty (counts and percentages)?

3. Which particular counties are experiencing the highest percentages of poverty?

### Install and Download

This project requires atleast Python 2.7 and the following Python libraries installed:

    pandas
    seaborn
	numpy
	matplotlib
	plotly
	plotly.figure_factory
	warnings
	
 Be sure to Install conda install -c conda-forge nodejs  
 
after installing nodejs,download Jupyter plotly extension: jupyter labextension install @jupyterlab/plotly-extension
   
  You will also need to have software installed to run and execute an iPython Notebook

I highly recommend using Anaconda, a pre-packaged Python distribution that contains all of the necessary libraries and software for this project. Jupyter is included and can run Ipython notebooks. 

 Please download these files from this repository: 
 Poverty in The United States_2017.ipynb-Overall poverty data including percentage and individual counts per state and county.
 Poverty States.xlsx-Aggregate state data for poverty individual counts and percentages
 
 The download from the census site is one file, I will disclose that I split the file into parts for ease of use. 

### Run

In the Anaconda Prompt, navigate to the folder the project files have been saved in: C:\Users\User\Location\Folder (Input your information, this is just an example). Once the location has been changed, type in the Anaconda Prompt:

    jupyter notebook Poverty in The United States_2017.ipynb

You may need to select a browser to view the notebook in such as Internet Explorer. This will open the Jupter Notebook software and project file in your browser.

### Results

States with highest poverty and hightes poverty counts differ. The states with high population density have a higher overall population count, while states with highest poverty percentages derive from states that are deeply poverty stricken and have overall population density. There are five South Dakota counties in the United States that are among the ten counties with the highest poverty percentages-Ziebach County, Todd County, Buffalo County, Corson County and Oglala Lakota County.

### Licensing, Authors, Acknowledgements
Thank you to The United States Census for provindg the data for this project. below are links to the data used. 

1.https://www.census.gov/data/tables/time-series/demo/income-poverty/historical-poverty-thresholds.html

2.http://www.unesco.org/new/en/social-and-human-sciences/themes/international-migration/glossary/poverty

3.http://www.worldometers.info/world-population/us-population/
