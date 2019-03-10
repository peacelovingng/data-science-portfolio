
# Combining, Cleaning and Normalizing Data
## Indian Health Service 

#### Tribes
According to the National Congress of American Indians, there are approximately 573 federally recognized Indian Nations (interchangeably called tribes, nations, bands, pueblos, communities and native villages) in the United States <sup> 1</sup>. Nearly 40% (229) are located in the state of Alaska, while the others are located in 35 other states. Each Nation is ethnically, culturally and linguistically diverse with its own unique history<sup>1</sup>. 

#### Indian Health Service
The goal of the Indian Health Service (IHS), is to provide federally funded health service to American Indians and/or Alaska Natives. The National Patient Information Reporting System (NPIRS), instituted the National Data Warehouse (NDW). The NDW is a data warehouse environment specifically for the Indian Health Service's (IHS) national data repository<sup> 2</sup>. Within this repository is information regarding various levels of patient care. https://www.ihs.gov/ndw/

#### Standard Codebook
Looking deeper within the NDW are tables with the approved codes sets from the Indian Health Service (IHS) Standard Code Book (SCB). The Standard Code Book is a uniform listing of descriptive terms and codes for the purpose of recording/reporting medical information collected during the provision of health care services. One of the tables is named "Community". A community is an area in which a tribe, is known to reside. It is not designated by coordinates. Because of this, it is difficult to determine where a community may be. 

#### Goal of Project
The goal of this project is to associate Indian Health Service communities to a Geographic Names Information System (GNIS) location. The Indian Health Service (IHS) consists of 14,770 unique active and inactive communities. This project also demonstrates some of the most tedious part sof preparing datasets for exploration=data cleansing and normalization-also called munging. 

## Geographic Information Names System (GNIS)
The Geographic Names Information System (GNIS) is the federal and national standard for geographic nomenclature.<sup> 3</sup> The U.S. Geological Survey developed the GNIS in support of the U.S. Board on Geographic Names as the official repository of domestic geographic names data, the official vehicle for geographic names use by all departments of the Federal Government, and the source for applying geographic names to federal electronic and printed products <sup> 3</sup>. The GNIS download consists of 2,279,278 locations within the United States as a text file. There are also individual state text files for usage. 


### Datasets
Table 1: Geographic Names Information Systems (GNIS) columns

| Name                  | Type      | 
|-----------------------|-----------|
| Feature Name          | Number    |
| Feature Class         | Character |
| State Alpha           | Character |
| State Numeric         | Character |
| County Name           | Character |
| County Numeric        | Character |
| Primary Latitude DMS  | Character |
| Primary Longitude DMS | Character |
| Primary Latitude DEC  | Number    |
| Primary Longitude DEC | Number    |   
| Elevation (meters)    | Number    |
| Source Latitude DMS   | Character |
| Source Longitude DMS  | Character |
| Source Latitude DEC   | Number    |
| Source Longitude DEC  | Number    |                                 
| Elevation (feet)      | Number    |
| Map Name              | Character |
| Date Created          | Date      |
| Date Edited           | Date      |


Table 2: Community table from Indian Health Service Standard Codebook

| Name      | Type      |
|-----------|-----------|
| Code      | Number    |
| State     | Character |
| County    | Character |
| Community | Character |
| ASU Code  | Number    |
| Status    | Character |


### Download Files

A copy of The Indian Health Service communities was downloaded from the standard codebook in excel file format: https://www.ihs.gov/scb/index.cfm?module=W_COMMUNITY&option=list&num=84&newquery=1

As well as a copy of the GNIS locations in text file format (NationalFile_20181201.zip):
https://geonames.usgs.gov/domestic/download_data.htm 

## Basic Normalization
The data was normalized by converting the columns of interest to string format for easier data handling, stripped of leading and trailing whitespaces and applied uppercase lettering. Inactive communities and unknown locations were excluded from this analysis. The same process was completed for the GNIS file for columns: ‘COUNTY_NUMERIC’, ‘STATE_NUMERIC’, ‘FEATURE_NAME’ and ‘FEATURE_CLASS’. The assumption is the Community column in the IHS standard codebook should be similar or identical to the FEATURE_NAME column in the GNIS dataset.


```python
import pandas as pd  
import fuzzymatcher  
#Read in the IHS Communities File
df1=pd.read_excel('Communities.xlsx')
#Change Columns of interest to strings
df1[['Code', 'State', 'County', 'Community', 'Status']] = df1[['Code', 'State', 'County', 'Community', 'Status']].astype(str)
#Apply uppercase and strip leading and trailing whitespaces of all object columns
df1 = df1.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
df1 = df1.apply(lambda x: x.str.upper() if x.dtype == "object" else x)
#The code is the INCITS code, I am only interested in the state code at this time
df1['Code'] = df1['Code'].str[0:2]
#Checks to view result and counts...a must
df1.head() 
#Checking the overall structure of the dataset
df1.shape
```




    (14769, 6)




```python
#Read in the GNIS file
df2=pd.read_csv('NationalFile_20181201.txt', "|")
#Change Columns of interest to strings
df2[['COUNTY_NUMERIC', 'STATE_NUMERIC', 'FEATURE_NAME', 'FEATURE_CLASS', 'STATE_ALPHA', 'COUNTY_NAME']] = df2[['COUNTY_NUMERIC', 'STATE_NUMERIC', 'FEATURE_NAME', 'FEATURE_CLASS', 'STATE_ALPHA', 'COUNTY_NAME']].astype(str)
#Apply uppercase and strip leading and trailing whitespaces of all object columns
df2 = df2.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
df2 = df2.apply(lambda x: x.str.upper() if x.dtype == "object" else x)
#Checks to view result and counts...checking the FEATURE_NAME column given that it is of most interest at this time
df2['FEATURE_NAME'] 
df2.head() 
#Looking for Unnamed Columns
df2.loc[:,~df2.columns.str.contains('^Unnamed')]  
#Checking the overall structure of the dataset
df2.shape
```




    (2279278, 20)



Inner Merge
Once the data in both datasets were normalized, an inner merge was performed on State Code (Code), Community, STATE_NUMERIC and FEATURE_NAME and saved in a dataframe (df3). After the merge, duplicates were dropped based upon state and Community name. Then communities were identified that were not included in this merge due to spelling differences between IHS and GNIS datasets. The merged output was then placed into a separate dataframe (df4). 


## Merging/Matching


```python
#Merging IHS dataset to GNIS dataset based on the two digit State Code and Name
df3=pd.merge(df1,df2, how='inner', left_on=['Code','Community'],right_on=['STATE_NUMERIC','FEATURE_NAME'])  
df3.shape  
#Drop duplicates after initial merge (there are many locations in the United States that have identical names, 
#only interested in locations within the same state as a quality check)
df4=df3.sort_values('Community').drop_duplicates(subset=['Code','Community'],keep='first')  
df4.loc[:,~df4.columns.str.contains('^Unnamed')]  
df4.shape 
```




    (9734, 26)



#### Identitfy Residuals

The communities identified that were not included in the merge were classified as “residuals”.  These residuals were placed into a dataframe (df5) to isolate them from the rest of the data to be used for further analysis. 


```python
df5=df1[~df1['Community'].isin(df4['FEATURE_NAME'])]  
df5.shape  
```




    (4055, 6)




```python
df5.head()
```




    <bound method NDFrame.head of       Code State              County           Community ASU Code    Status
    0       01    AL             BALDWIN         BAY MINETTE     5874    ACTIVE
    1       01    AL             BALDWIN              DAPHNE     5874    ACTIVE
    3       01    AL             BALDWIN            FAIRHOPE     5874    ACTIVE
    5       01    AL             BALDWIN         ROBERTSDALE     5874    ACTIVE
    7       01    AL             BALDWIN          SUMMERDALE     5874    ACTIVE
    8       01    AL             BALDWIN          SILVERHILL     5874    ACTIVE
    9       01    AL             BALDWIN             PERDIDO     5874    ACTIVE
    10      01    AL             BALDWIN              LOXLEY     5874    ACTIVE
    11      01    AL             BALDWIN         GULF SHORES     5874    ACTIVE
    12      01    AL             BALDWIN        ORANGE BEACH     5874    ACTIVE
    14      01    AL             BALDWIN             LILLIAN     5874    ACTIVE
    15      01    AL             BALDWIN        SPANISH FORT     5874    ACTIVE
    17      01    AL             BALDWIN           STAPLETON     5874    ACTIVE
    18      01    AL              BUTLER            MCKENZIE     5800    ACTIVE
    20      01    AL             CONECUH              REPTON     5800    ACTIVE
    21      01    AL              DEKALB          RAINSVILLE     5800    ACTIVE
    22      01    AL            ESCAMBIA             BREWTON     5874    ACTIVE
    23      01    AL            ESCAMBIA       EAST ESCAMBIA     5874    ACTIVE
    24      01    AL            ESCAMBIA            FLOMATON     5874    ACTIVE
    25      01    AL            ESCAMBIA        MCCULLOUG-HU     5874    ACTIVE
    26      01    AL            ESCAMBIA             HUXFORD     5874    ACTIVE
    27      01    AL            ESCAMBIA        EAST BREWTON     5874    ACTIVE
    28      01    AL            ESCAMBIA              ATMORE     5874    ACTIVE
    29      01    AL              MOBILE        BAYOU LA BAT     5874    ACTIVE
    30      01    AL              MOBILE          CITRONELLA     5874    ACTIVE
    31      01    AL              MOBILE           GRAND BAY     5874    ACTIVE
    32      01    AL              MOBILE              MOBILE     5874    ACTIVE
    34      01    AL              MOBILE              SEMMES     5874    ACTIVE
    35      01    AL              MOBILE        TANNER-WILLM     5874    ACTIVE
    36      01    AL              MOBILE            THEODORE     5874    ACTIVE
    ...    ...   ...                 ...                 ...      ...       ...
    14679   55    WI   WISCONSIN UNKNOWN        WISCONSN UNK     1100  INACTIVE
    14680   55    WI   WISCONSIN UNKNOWN        WISCONSN UNK     1100  INACTIVE
    14694   56    WY             FREMONT        BOULDER FLAT     4046    ACTIVE
    14697   56    WY             FREMONT        DRY CREEK RA     4046    ACTIVE
    14700   56    WY             FREMONT         FT WASHAKIE     4046    ACTIVE
    14708   56    WY             FREMONT         ST STEPHENS     4046    ACTIVE
    14714   56    WY         HOT SPRINGS        HAMILTON DOM     4046    ACTIVE
    14722   56    WY             NATRONA             BARNUNN     4000    ACTIVE
    14746   56    WY         WYOMING UNK         WYOMING UNK     4000    ACTIVE
    14747   56    WY         WYOMING UNK         WYOMING UNK     4000  INACTIVE
    14748   56    WY         WYOMING UNK         WYOMING UNK     4000  INACTIVE
    14750   72    PR     PUERTO RICO UNK     PUERTO RICO UNK     9999    ACTIVE
    14751   72    PR     PUERTO RICO UNK        PUERTO R UNK     5800  INACTIVE
    14752   72    PR     PUERTO RICO UNK        PUERTO R UNK     5800  INACTIVE
    14753   78    VI  VIRGIN ISLANDS UNK  VIRGIN ISLANDS UNK     9999    ACTIVE
    14754   78    VI  VIRGIN ISLANDS UNK        VIRGIN I UNK     5800  INACTIVE
    14755   78    VI  VIRGIN ISLANDS UNK        VIRGIN I UNK     5800  INACTIVE
    14756   95    NL     NETHERLANDS UNK     NETHERLANDS UNK     9999    ACTIVE
    14757   96    CA          CANADA UNK            CORNWALL     5800    ACTIVE
    14758   96    CA          CANADA UNK     CORNWALL ISLAND     5800    ACTIVE
    14759   96    CA          CANADA UNK                SNYE     5800    ACTIVE
    14760   96    CA          CANADA UNK           ST. REGIS     5800    ACTIVE
    14761   96    CA          CANADA UNK          CANADA UNK     9999    ACTIVE
    14762   96    CA          CANADA UNK          CANADA UNK     9999  INACTIVE
    14763   96    CA          CANADA UNK          CANADA UNK     9999  INACTIVE
    14764   97    MX          MEXICO UNK          MEXICO UNK     9999    ACTIVE
    14765   97    MX          MEXICO UNK          MEXICO UNK     9999  INACTIVE
    14766   97    MX          MEXICO UNK          MEXICO UNK     9999  INACTIVE
    14767   98   NAN         UNSPECIFIED         UNSPECIFIED     9999  INACTIVE
    14768   99     ?             UNKNOWN             UNKNOWN     9999    ACTIVE
    
    [4055 rows x 6 columns]>



#### Fuzzy Matching
A fuzzy match was performed on the residuals with a python package named “fuzzymatcher”, and saved in a dataframe (df6).The goal of this package is to match two dataframes based upon one or two similar fields. A fuzzy match was performed on Community and FEATURE_NAME. After matches are identified, the package ranks each record by probabilistic scoring. Duplicate records were excluded by both state and Community name so that only unique matches by state were included in the final output. Any excluded community was classified as a residual. 
This is the ink to Github repository for the fuzzymatcher package: https://github.com/RobinL/fuzzymatcher



```python
#This will take some time to run-not the most efficient method on a dataset this size
df6=fuzzymatcher.fuzzy_left_join(df5,df2,left_on="Community", right_on="FEATURE_NAME")  
df6.shape 
```




    (4055, 29)




```python
df6.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>best_match_score</th>
      <th>__id_left</th>
      <th>__id_right</th>
      <th>Code</th>
      <th>State</th>
      <th>County</th>
      <th>Community</th>
      <th>ASU Code</th>
      <th>Status</th>
      <th>FEATURE_ID</th>
      <th>...</th>
      <th>PRIM_LONG_DEC</th>
      <th>SOURCE_LAT_DMS</th>
      <th>SOURCE_LONG_DMS</th>
      <th>SOURCE_LAT_DEC</th>
      <th>SOURCE_LONG_DEC</th>
      <th>ELEV_IN_M</th>
      <th>ELEV_IN_FT</th>
      <th>MAP_NAME</th>
      <th>DATE_CREATED</th>
      <th>DATE_EDITED</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.437734</td>
      <td>0_left</td>
      <td>78409_right</td>
      <td>01</td>
      <td>AL</td>
      <td>BALDWIN</td>
      <td>BAY MINETTE</td>
      <td>5874</td>
      <td>ACTIVE</td>
      <td>113588.0</td>
      <td>...</td>
      <td>-87.773047</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>82.0</td>
      <td>269.0</td>
      <td>BAY MINETTE NORTH</td>
      <td>09/04/1980</td>
      <td>08/16/2013</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0.315869</td>
      <td>1_left</td>
      <td>121385_right</td>
      <td>01</td>
      <td>AL</td>
      <td>BALDWIN</td>
      <td>DAPHNE</td>
      <td>5874</td>
      <td>ACTIVE</td>
      <td>157933.0</td>
      <td>...</td>
      <td>-87.903605</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>49.0</td>
      <td>161.0</td>
      <td>DAPHNE</td>
      <td>09/04/1980</td>
      <td>12/06/2013</td>
    </tr>
    <tr>
      <th>51</th>
      <td>0.323311</td>
      <td>2_left</td>
      <td>82920_right</td>
      <td>01</td>
      <td>AL</td>
      <td>BALDWIN</td>
      <td>FAIRHOPE</td>
      <td>5874</td>
      <td>ACTIVE</td>
      <td>118120.0</td>
      <td>...</td>
      <td>-87.903326</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>37.0</td>
      <td>121.0</td>
      <td>DAPHNE</td>
      <td>09/04/1980</td>
      <td>08/27/2013</td>
    </tr>
    <tr>
      <th>101</th>
      <td>0.368564</td>
      <td>3_left</td>
      <td>90480_right</td>
      <td>01</td>
      <td>AL</td>
      <td>BALDWIN</td>
      <td>ROBERTSDALE</td>
      <td>5874</td>
      <td>ACTIVE</td>
      <td>125703.0</td>
      <td>...</td>
      <td>-87.711932</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>44.0</td>
      <td>144.0</td>
      <td>ROBERTSDALE</td>
      <td>09/04/1980</td>
      <td>03/20/2008</td>
    </tr>
    <tr>
      <th>122</th>
      <td>0.358645</td>
      <td>4_left</td>
      <td>118725_right</td>
      <td>01</td>
      <td>AL</td>
      <td>BALDWIN</td>
      <td>SUMMERDALE</td>
      <td>5874</td>
      <td>ACTIVE</td>
      <td>155262.0</td>
      <td>...</td>
      <td>-87.699709</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>34.0</td>
      <td>112.0</td>
      <td>FOLEY</td>
      <td>09/04/1980</td>
      <td>03/20/2008</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 29 columns</p>
</div>




```python
#Keeping only state to state output
df7 = df6[df6['State'] == df6['STATE_ALPHA']]  
df7.shape  
#Save only unique output and dropping duplicates
df8=df7.sort_values('Community').drop_duplicates(subset=['Community','Code'],keep='first')  
df8.shape  
```




    (1884, 29)




```python
df8.columns
```




    Index(['best_match_score', '__id_left', '__id_right', 'Code', 'State',
           'County', 'Community', 'ASU Code', 'Status', 'FEATURE_ID',
           'FEATURE_NAME', 'FEATURE_CLASS', 'STATE_ALPHA', 'STATE_NUMERIC',
           'COUNTY_NAME', 'COUNTY_NUMERIC', 'PRIMARY_LAT_DMS', 'PRIM_LONG_DMS',
           'PRIM_LAT_DEC', 'PRIM_LONG_DEC', 'SOURCE_LAT_DMS', 'SOURCE_LONG_DMS',
           'SOURCE_LAT_DEC', 'SOURCE_LONG_DEC', 'ELEV_IN_M', 'ELEV_IN_FT',
           'MAP_NAME', 'DATE_CREATED', 'DATE_EDITED'],
          dtype='object')



The output after the fuzzy matching contains columns I do not want, so I simply deleted them. This was useful if I plan to do further matching in another method and would like to output my results to another file.


```python
#Need to see what columns are included in this dataframe before needing to delete
df8.drop(['best_match_score','__id_left','__id_right'], axis=1, inplace=True)
#Saving the results from initial match to a file
df8.to_excel('Fuzzy Communities With Match_1.xlsx') 
```

#### Identify Further Residuals
Locations that were not included in the first iteration of matches are to be identified here. Further matching on text is necessary to ensure the most locations are captured. 


```python
df9=df6[df6['State'] != df6['STATE_ALPHA']]  
df10=df9.sort_values('Community').drop_duplicates(subset=['Community','Code'],keep='first')  
```


```python
df10.shape
```




    (2035, 29)



## Conclusion

In conclusion 2,035 IHS locations remain unlocated due to various reasons due to differences in spelling, the location existing in a neighboring state or simply not existing in both datasets. Further discovery is needed from this point forward. I hope you found this notebook useful especially when dealing with text data and joining two textual datasets. 

1. Tribal Nations & the United States: An Introduction. Retrieved from http://www.ncai.org/about-tribeshttp://www.ncai.org/about-tribes. Accessed March 7, 2019

2. Indian Health Service. The Federal Health Program for American Indians and Alaska Natives. https://www.ihs.gov/scb/index.cfm?module=W_COMMUNITY&option=list&num=84&newquery=1. Accessed March 7, 2019.

3. Eastern Region Geography. Domestic and Antarctic Names - State and Topical Gazetteer Download Files. https://geonames.usgs.gov/domestic/download_data.htm. Accessed March 7, 2019. 

