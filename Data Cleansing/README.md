# Combining, Cleaning and Normalizing Data

This is a basic data cleansing project to demonstrate the steps often times necessary before conducting analysis on textual data. Data was downloaded from The Indian Health Service NDW as well as The Geographic Names Information System (GNIS) to determine a standardized location for American Indian Nations in the United States. 

### Install and Download

This project requires atleast Python 2.7 and the following Python libraries installed:

    Pandas
    fuzzymatcher
	
 Note: fuzzymatcher utilizes the sqlite's full text search to find matches. If deciding to use the fuzzymatcher package, you may need to update the sqlite3 DLL file located in your Anaconda or Python folder.    
   
  
You will also need to have software installed to run and execute an iPython Notebook

I highly recommend using Anaconda, a pre-packaged Python distribution that contains all of the necessary libraries and software for this project. Jupyter is included and can run Ipython notebooks. 

 Please download these files from this repository: Data Cleansing Project.ipynb, and Communities.xlsx
 NationalFile_20181201 needs to be downloaded from the GNIS website (it was too large to upload to Github)- https://geonames.usgs.gov/domestic/download_data.htm

### Run

In the Anaconda Prompt, navigate to the folder the project files have been saved in: C:\Users\User\Location\Folder (Input your information, this is just an example). Once the location has been changed, type in the Anaconda Prompt:

    jupyter notebook Data Cleansing Project.ipynb

You may need to select a browser to view the notebook in such as Internet Explorer. This will open the Jupter Notebook software and project file in your browser.
