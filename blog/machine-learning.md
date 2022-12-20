<div class="cell markdown" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2022-12-20T10:45:34.435300&quot;,&quot;end_time&quot;:&quot;2022-12-20T10:45:34.452031&quot;,&quot;duration&quot;:1.6731e-2}" data-tags="[]">

# Weather Prediction

</div>

<div class="cell markdown" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2022-12-20T10:45:34.466351&quot;,&quot;end_time&quot;:&quot;2022-12-20T10:45:34.482053&quot;,&quot;duration&quot;:1.5702e-2}" data-tags="[]">

We have a sample weather dataset of JFK airport available since 1970.
Some objectives include:

  - Cleaning the data
  - Train a machine learning model to make historical and future
    predictions.

</div>

<div class="cell markdown" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2022-12-20T10:45:34.495997&quot;,&quot;end_time&quot;:&quot;2022-12-20T10:45:34.511071&quot;,&quot;duration&quot;:1.5074e-2}" data-tags="[]">

Import the data with the `DATE` column as index. `DATE` column  gives
each row a unique identifier so each row is is going to be referred to
by date.

</div>

<div class="cell code" data-execution_count="1" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-12-20T10:45:34.791937Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-12-20T10:45:34.549746Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-12-20T10:45:34.793453Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-12-20T10:45:34.549041Z&quot;}" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2022-12-20T10:45:34.527294&quot;,&quot;end_time&quot;:&quot;2022-12-20T10:45:34.798421&quot;,&quot;duration&quot;:0.271127}" data-tags="[]">

``` python
import pandas as pd

weather = pd.read_csv("/kaggle/input/jfk-weather/jfk_weather.csv", index_col="DATE")

weather.head()
```

<div class="output execute_result" data-execution_count="1">

``` 
                STATION                              NAME  ACMH   ACSH  AWND  \
DATE                                                                           
01-01-1970  USW00094789  JFK INTERNATIONAL AIRPORT, NY US  80.0   90.0   NaN   
02-01-1970  USW00094789  JFK INTERNATIONAL AIRPORT, NY US  30.0   20.0   NaN   
03-01-1970  USW00094789  JFK INTERNATIONAL AIRPORT, NY US  80.0  100.0   NaN   
04-01-1970  USW00094789  JFK INTERNATIONAL AIRPORT, NY US  10.0   20.0   NaN   
05-01-1970  USW00094789  JFK INTERNATIONAL AIRPORT, NY US  30.0   10.0   NaN   

            FMTM  PGTM  PRCP  SNOW  SNWD  ...  WT11  WT13  WT14  WT15  WT16  \
DATE                                      ...                                 
01-01-1970   NaN   NaN  0.00   0.0   0.0  ...   NaN   NaN   NaN   NaN   NaN   
02-01-1970   NaN   NaN  0.00   0.0   0.0  ...   NaN   NaN   NaN   NaN   NaN   
03-01-1970   NaN   NaN  0.02   0.0   0.0  ...   NaN   NaN   NaN   NaN   1.0   
04-01-1970   NaN   NaN  0.00   0.0   0.0  ...   NaN   NaN   NaN   NaN   NaN   
05-01-1970   NaN   NaN  0.00   0.0   0.0  ...   NaN   NaN   NaN   NaN   NaN   

            WT17  WT18  WT21  WT22  WV01  
DATE                                      
01-01-1970   NaN   NaN   NaN   NaN   NaN  
02-01-1970   NaN   NaN   NaN   NaN   NaN  
03-01-1970   NaN   1.0   NaN   NaN   NaN  
04-01-1970   NaN   1.0   NaN   NaN   NaN  
05-01-1970   NaN   NaN   NaN   NaN   NaN  

[5 rows x 44 columns]
```

</div>

</div>

<div class="cell markdown" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2022-12-20T10:45:34.820827&quot;,&quot;end_time&quot;:&quot;2022-12-20T10:45:34.837823&quot;,&quot;duration&quot;:1.6996e-2}" data-tags="[]">

You might see a lot of `Nan` values. Machine learning models will
not work with missing values. So the first data cleaning needs to be
done. We can get rid of our columns that have missing values. We can
calculate the null percentage in the data which basically applies a
function `pd.isnull` the weather data frame.

</div>

<div class="cell code" data-execution_count="2" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-12-20T10:45:34.892977Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-12-20T10:45:34.871357Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-12-20T10:45:34.894529Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-12-20T10:45:34.870914Z&quot;}" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2022-12-20T10:45:34.854689&quot;,&quot;end_time&quot;:&quot;2022-12-20T10:45:34.897025&quot;,&quot;duration&quot;:4.2336e-2}" data-tags="[]">

``` python
null_pct = weather.apply(pd.isnull).sum()/weather.shape[0]  #sum of missing values in the column by divided by the number of rows
null_pct
```

<div class="output execute_result" data-execution_count="2">

    STATION    0.000000
    NAME       0.000000
    ACMH       0.502715
    ACSH       0.502664
    AWND       0.264650
    FMTM       0.476390
    PGTM       0.365451
    PRCP       0.000000
    SNOW       0.000000
    SNWD       0.000052
    TAVG       0.678769
    TMAX       0.000000
    TMIN       0.000000
    TSUN       0.998397
    WDF1       0.502922
    WDF2       0.497492
    WDF5       0.501784
    WDFG       0.735144
    WDFM       0.999948
    WESD       0.686010
    WSF1       0.502767
    WSF2       0.497492
    WSF5       0.501836
    WSFG       0.614016
    WSFM       0.999948
    WT01       0.630308
    WT02       0.935040
    WT03       0.933437
    WT04       0.982622
    WT05       0.981174
    WT06       0.990639
    WT07       0.994414
    WT08       0.797259
    WT09       0.992759
    WT11       0.999276
    WT13       0.886993
    WT14       0.954125
    WT15       0.997828
    WT16       0.659840
    WT17       0.996897
    WT18       0.939643
    WT21       0.999741
    WT22       0.997466
    WV01       0.999948
    dtype: float64

</div>

</div>

<div class="cell markdown" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2022-12-20T10:45:34.912260&quot;,&quot;end_time&quot;:&quot;2022-12-20T10:45:34.926539&quot;,&quot;duration&quot;:1.4279e-2}" data-tags="[]">

Some of the reasons for missing data can be due to malfunctioning errors
and human errors while noting down the data. We can clean up the data by
removing columns that have a null percentage less than 5% and we'll
index those columns in a new column called `valid_columns`.

</div>

<div class="cell code" data-execution_count="3" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-12-20T10:45:34.969393Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-12-20T10:45:34.960653Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-12-20T10:45:34.970736Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-12-20T10:45:34.957847Z&quot;}" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2022-12-20T10:45:34.941066&quot;,&quot;end_time&quot;:&quot;2022-12-20T10:45:34.974176&quot;,&quot;duration&quot;:3.311e-2}" data-tags="[]">

``` python
valid_columns = weather.columns[null_pct < .05]
valid_columns
```

<div class="output execute_result" data-execution_count="3">

    Index(['STATION', 'NAME', 'PRCP', 'SNOW', 'SNWD', 'TMAX', 'TMIN'], dtype='object')

</div>

</div>

<div class="cell markdown" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2022-12-20T10:45:34.992182&quot;,&quot;end_time&quot;:&quot;2022-12-20T10:45:35.010401&quot;,&quot;duration&quot;:1.8219e-2}" data-tags="[]">

In order to avoid getting an error later, we can assign the data frame a
slice of the data frame back itself by using the `.copy()` method.

</div>

<div class="cell code" data-execution_count="4" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-12-20T10:45:35.066484Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-12-20T10:45:35.041851Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-12-20T10:45:35.067773Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-12-20T10:45:35.041349Z&quot;}" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2022-12-20T10:45:35.025183&quot;,&quot;end_time&quot;:&quot;2022-12-20T10:45:35.069953&quot;,&quot;duration&quot;:4.477e-2}" data-tags="[]">

``` python
weather = weather[valid_columns].copy()
weather.columns = weather.columns.str.lower()  #lower case the column names
weather
```

<div class="output execute_result" data-execution_count="4">

``` 
                station                              name  prcp  snow  snwd  \
DATE                                                                          
01-01-1970  USW00094789  JFK INTERNATIONAL AIRPORT, NY US  0.00   0.0   0.0   
02-01-1970  USW00094789  JFK INTERNATIONAL AIRPORT, NY US  0.00   0.0   0.0   
03-01-1970  USW00094789  JFK INTERNATIONAL AIRPORT, NY US  0.02   0.0   0.0   
04-01-1970  USW00094789  JFK INTERNATIONAL AIRPORT, NY US  0.00   0.0   0.0   
05-01-1970  USW00094789  JFK INTERNATIONAL AIRPORT, NY US  0.00   0.0   0.0   
...                 ...                               ...   ...   ...   ...   
04-12-2022  USW00094789  JFK INTERNATIONAL AIRPORT, NY US  0.00   0.0   0.0   
05-12-2022  USW00094789  JFK INTERNATIONAL AIRPORT, NY US  0.00   0.0   0.0   
06-12-2022  USW00094789  JFK INTERNATIONAL AIRPORT, NY US  0.43   0.0   0.0   
07-12-2022  USW00094789  JFK INTERNATIONAL AIRPORT, NY US  0.31   0.0   0.0   
08-12-2022  USW00094789  JFK INTERNATIONAL AIRPORT, NY US  0.00   0.0   0.0   

            tmax  tmin  
DATE                    
01-01-1970    28    22  
02-01-1970    31    22  
03-01-1970    38    25  
04-01-1970    31    23  
05-01-1970    35    21  
...          ...   ...  
04-12-2022    47    37  
05-12-2022    47    30  
06-12-2022    57    36  
07-12-2022    57    54  
08-12-2022    55    42  

[19335 rows x 7 columns]
```

</div>

</div>

<div class="cell markdown" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2022-12-20T10:45:35.084636&quot;,&quot;end_time&quot;:&quot;2022-12-20T10:45:35.098979&quot;,&quot;duration&quot;:1.4343e-2}" data-tags="[]">

We can forward fill the missing snow depth using the last value of snow
depth because it makes sense that yesterday's snow depth will be equal
to today's snow depth.

</div>

<div class="cell code" data-execution_count="5" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-12-20T10:45:35.141730Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-12-20T10:45:35.131090Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-12-20T10:45:35.142751Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-12-20T10:45:35.129999Z&quot;}" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2022-12-20T10:45:35.113721&quot;,&quot;end_time&quot;:&quot;2022-12-20T10:45:35.145260&quot;,&quot;duration&quot;:3.1539e-2}" data-tags="[]">

``` python
weather = weather.ffill()
```

</div>

<div class="cell markdown" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2022-12-20T10:45:35.160459&quot;,&quot;end_time&quot;:&quot;2022-12-20T10:45:35.174972&quot;,&quot;duration&quot;:1.4513e-2}" data-tags="[]">

Let's look at how many missing values there are in each column

</div>

<div class="cell code" data-execution_count="6" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-12-20T10:45:35.219310Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-12-20T10:45:35.207447Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-12-20T10:45:35.220538Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-12-20T10:45:35.206132Z&quot;}" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2022-12-20T10:45:35.189631&quot;,&quot;end_time&quot;:&quot;2022-12-20T10:45:35.223232&quot;,&quot;duration&quot;:3.3601e-2}" data-tags="[]">

``` python
weather.apply(pd.isnull).sum()
```

<div class="output execute_result" data-execution_count="6">

    station    0
    name       0
    prcp       0
    snow       0
    snwd       0
    tmax       0
    tmin       0
    dtype: int64

</div>

</div>

<div class="cell markdown" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2022-12-20T10:45:35.238470&quot;,&quot;end_time&quot;:&quot;2022-12-20T10:45:35.253025&quot;,&quot;duration&quot;:1.4555e-2}" data-tags="[]">

We can see none of our columns have missing values now which is
perfect that's what we need for machine learning.

</div>

<div class="cell markdown" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2022-12-20T10:45:35.267865&quot;,&quot;end_time&quot;:&quot;2022-12-20T10:45:35.282279&quot;,&quot;duration&quot;:1.4414e-2}" data-tags="[]">

The next thing required for maching learning is that the data in the
columns are the correct data types.

</div>

<div class="cell code" data-execution_count="7" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-12-20T10:45:35.320833Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-12-20T10:45:35.314113Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-12-20T10:45:35.321900Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-12-20T10:45:35.313625Z&quot;}" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2022-12-20T10:45:35.297151&quot;,&quot;end_time&quot;:&quot;2022-12-20T10:45:35.324349&quot;,&quot;duration&quot;:2.7198e-2}" data-tags="[]">

``` python
weather.dtypes
```

<div class="output execute_result" data-execution_count="7">

    station     object
    name        object
    prcp       float64
    snow       float64
    snwd       float64
    tmax         int64
    tmin         int64
    dtype: object

</div>

</div>

<div class="cell markdown" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2022-12-20T10:45:35.339479&quot;,&quot;end_time&quot;:&quot;2022-12-20T10:45:35.354170&quot;,&quot;duration&quot;:1.4691e-2}" data-tags="[]">

So pandas columns can be of different data types. An `object` data type
usually indicates that the column is a string and in this case station,
name columns are a string. So these two columns are are stored as
objects and that's actually correct. Sometimes columns are incorrectly
sorted as objects and we will have to convert them to a numeric type. In
this case we don't and we can see that these other columns are also
stored as the correct type,

</div>

<div class="cell markdown" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2022-12-20T10:45:35.369090&quot;,&quot;end_time&quot;:&quot;2022-12-20T10:45:35.383695&quot;,&quot;duration&quot;:1.4605e-2}" data-tags="[]">

We also want to check our index which are those row labels to make sure
it's the correct type.

</div>

<div class="cell code" data-execution_count="8" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-12-20T10:45:35.421240Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-12-20T10:45:35.415741Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-12-20T10:45:35.422062Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-12-20T10:45:35.414936Z&quot;}" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2022-12-20T10:45:35.398451&quot;,&quot;end_time&quot;:&quot;2022-12-20T10:45:35.424188&quot;,&quot;duration&quot;:2.5737e-2}" data-tags="[]">

``` python
weather.index
```

<div class="output execute_result" data-execution_count="8">

    Index(['01-01-1970', '02-01-1970', '03-01-1970', '04-01-1970', '05-01-1970',
           '06-01-1970', '07-01-1970', '08-01-1970', '09-01-1970', '10-01-1970',
           ...
           '29-11-2022', '30-11-2022', '01-12-2022', '02-12-2022', '03-12-2022',
           '04-12-2022', '05-12-2022', '06-12-2022', '07-12-2022', '08-12-2022'],
          dtype='object', name='DATE', length=19335)

</div>

</div>

<div class="cell markdown" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2022-12-20T10:45:35.439365&quot;,&quot;end_time&quot;:&quot;2022-12-20T10:45:35.454093&quot;,&quot;duration&quot;:1.4728e-2}" data-tags="[]">

We can see that our index is stored as an object but it's actually a
date and if we convert it to a date it makes some of the processing
we're going to do later easier. Let's convert our index to a date time -
we'll take `weather.index` and then we're going to use the
`pandas.to_datetime` function to actually convert our index into a
`datetime`. We pass in our index and it's going to convert it from an
`object` into a `datetime`.

</div>

<div class="cell code" data-execution_count="9" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-12-20T10:45:35.511428Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-12-20T10:45:35.486802Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-12-20T10:45:35.512333Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-12-20T10:45:35.485994Z&quot;}" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2022-12-20T10:45:35.469130&quot;,&quot;end_time&quot;:&quot;2022-12-20T10:45:35.514396&quot;,&quot;duration&quot;:4.5266e-2}" data-tags="[]">

``` python
weather.index = pd.to_datetime(weather.index)
weather.index
```

<div class="output execute_result" data-execution_count="9">

    DatetimeIndex(['1970-01-01', '1970-02-01', '1970-03-01', '1970-04-01',
                   '1970-05-01', '1970-06-01', '1970-07-01', '1970-08-01',
                   '1970-09-01', '1970-10-01',
                   ...
                   '2022-11-29', '2022-11-30', '2022-01-12', '2022-02-12',
                   '2022-03-12', '2022-04-12', '2022-05-12', '2022-06-12',
                   '2022-07-12', '2022-08-12'],
                  dtype='datetime64[ns]', name='DATE', length=19335, freq=None)

</div>

</div>

<div class="cell markdown" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2022-12-20T10:45:35.529786&quot;,&quot;end_time&quot;:&quot;2022-12-20T10:45:35.544978&quot;,&quot;duration&quot;:1.5192e-2}" data-tags="[]">

On looking at the index again and we can see it's now stored as a
`datetime`. It doesn't look any different but how pandas is handling
it internally is different. We can now perform useful operations like
getting the year component of the date.

</div>

<div class="cell code" data-execution_count="10" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-12-20T10:45:35.586398Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-12-20T10:45:35.578034Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-12-20T10:45:35.587604Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-12-20T10:45:35.577223Z&quot;}" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2022-12-20T10:45:35.560141&quot;,&quot;end_time&quot;:&quot;2022-12-20T10:45:35.590026&quot;,&quot;duration&quot;:2.9885e-2}" data-tags="[]">

``` python
weather.index.year
```

<div class="output execute_result" data-execution_count="10">

    Int64Index([1970, 1970, 1970, 1970, 1970, 1970, 1970, 1970, 1970, 1970,
                ...
                2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022],
               dtype='int64', name='DATE', length=19335)

</div>

</div>

<div class="cell markdown" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2022-12-20T10:45:35.606706&quot;,&quot;end_time&quot;:&quot;2022-12-20T10:45:35.622145&quot;,&quot;duration&quot;:1.5439e-2}" data-tags="[]">

One last just check of our data before we continue and actually jump
into machine learning is we're going to make sure that we don't have any
gaps in our data. To do this we're going to count up how many rows we
have for each year and then we're going to sort that by our our year. So
`value_counts` will count up how many times each unique value occurs
here.

</div>

<div class="cell code" data-execution_count="11" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-12-20T10:45:35.671755Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-12-20T10:45:35.656019Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-12-20T10:45:35.672945Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-12-20T10:45:35.655262Z&quot;}" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2022-12-20T10:45:35.637998&quot;,&quot;end_time&quot;:&quot;2022-12-20T10:45:35.675469&quot;,&quot;duration&quot;:3.7471e-2}" data-tags="[]">

``` python
weather.index.year.value_counts().sort_index()
```

<div class="output execute_result" data-execution_count="11">

    1970    365
    1971    365
    1972    366
    1973    365
    1974    365
    1975    365
    1976    366
    1977    365
    1978    365
    1979    365
    1980    366
    1981    365
    1982    365
    1983    365
    1984    366
    1985    365
    1986    365
    1987    365
    1988    366
    1989    365
    1990    365
    1991    365
    1992    366
    1993    365
    1994    365
    1995    365
    1996    366
    1997    365
    1998    365
    1999    365
    2000    366
    2001    365
    2002    365
    2003    365
    2004    366
    2005    365
    2006    365
    2007    365
    2008    366
    2009    365
    2010    365
    2011    365
    2012    366
    2013    365
    2014    365
    2015    365
    2016    366
    2017    365
    2018    365
    2019    365
    2020    366
    2021    365
    2022    342
    Name: DATE, dtype: int64

</div>

</div>

<div class="cell markdown" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2022-12-20T10:45:35.691947&quot;,&quot;end_time&quot;:&quot;2022-12-20T10:45:35.707641&quot;,&quot;duration&quot;:1.5694e-2}" data-tags="[]">

It doesn't look like we have any gaps or other issues. If you used your
local weather data you may actually have some gaps and in which
case it's usually okay if you have some gaps. But just something to
keep in mind if you have too many gaps you may not be able to make any
predictions. Just to take another look at gaps and make sure there's
nothing weird going on we can actually plot some of our columns. So this
is the snow depth column so that's how much snow has accumulated on the
ground for that day and then we can use the `.plot()` method to
actually create a bar plot showing snow depth by day.

</div>

<div class="cell code" data-execution_count="12" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-12-20T10:45:36.162098Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-12-20T10:45:35.741658Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-12-20T10:45:36.163359Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-12-20T10:45:35.740928Z&quot;}" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2022-12-20T10:45:35.723408&quot;,&quot;end_time&quot;:&quot;2022-12-20T10:45:36.165886&quot;,&quot;duration&quot;:0.442478}" data-tags="[]">

``` python
weather["snwd"].plot()
```

<div class="output execute_result" data-execution_count="12">

    <AxesSubplot:xlabel='DATE'>

</div>

<div class="output display_data">

![](56e2441c5e9bf50c72b0f22141360f9d179fb035.png)

</div>

</div>

<div class="cell code" data-execution_count="13" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-12-20T10:45:36.219977Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-12-20T10:45:36.200919Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-12-20T10:45:36.221213Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-12-20T10:45:36.200126Z&quot;}" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2022-12-20T10:45:36.182110&quot;,&quot;end_time&quot;:&quot;2022-12-20T10:45:36.224202&quot;,&quot;duration&quot;:4.2092e-2}" data-tags="[]">

``` python
weather
```

<div class="output execute_result" data-execution_count="13">

``` 
                station                              name  prcp  snow  snwd  \
DATE                                                                          
1970-01-01  USW00094789  JFK INTERNATIONAL AIRPORT, NY US  0.00   0.0   0.0   
1970-02-01  USW00094789  JFK INTERNATIONAL AIRPORT, NY US  0.00   0.0   0.0   
1970-03-01  USW00094789  JFK INTERNATIONAL AIRPORT, NY US  0.02   0.0   0.0   
1970-04-01  USW00094789  JFK INTERNATIONAL AIRPORT, NY US  0.00   0.0   0.0   
1970-05-01  USW00094789  JFK INTERNATIONAL AIRPORT, NY US  0.00   0.0   0.0   
...                 ...                               ...   ...   ...   ...   
2022-04-12  USW00094789  JFK INTERNATIONAL AIRPORT, NY US  0.00   0.0   0.0   
2022-05-12  USW00094789  JFK INTERNATIONAL AIRPORT, NY US  0.00   0.0   0.0   
2022-06-12  USW00094789  JFK INTERNATIONAL AIRPORT, NY US  0.43   0.0   0.0   
2022-07-12  USW00094789  JFK INTERNATIONAL AIRPORT, NY US  0.31   0.0   0.0   
2022-08-12  USW00094789  JFK INTERNATIONAL AIRPORT, NY US  0.00   0.0   0.0   

            tmax  tmin  
DATE                    
1970-01-01    28    22  
1970-02-01    31    22  
1970-03-01    38    25  
1970-04-01    31    23  
1970-05-01    35    21  
...          ...   ...  
2022-04-12    47    37  
2022-05-12    47    30  
2022-06-12    57    36  
2022-07-12    57    54  
2022-08-12    55    42  

[19335 rows x 7 columns]
```

</div>

</div>

<div class="cell markdown" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2022-12-20T10:45:36.241261&quot;,&quot;end_time&quot;:&quot;2022-12-20T10:45:36.257724&quot;,&quot;duration&quot;:1.6463e-2}" data-tags="[]">

When we use a machine learning algorithm we need to tell the algorithm
what we're trying to predict and in our case what we're trying to
predict is tomorrow's temperature - `T-Max` using today's information.
We'll create a `target` column in our weather dataframe and use
`.shift()` method to keep the same index but it pulls the values from
the next row back.

</div>

<div class="cell code" data-execution_count="14" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-12-20T10:45:36.318096Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-12-20T10:45:36.293968Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-12-20T10:45:36.319258Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-12-20T10:45:36.293200Z&quot;}" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2022-12-20T10:45:36.274982&quot;,&quot;end_time&quot;:&quot;2022-12-20T10:45:36.321652&quot;,&quot;duration&quot;:4.667e-2}" data-tags="[]">

``` python
weather["target"] = weather.shift(-1)["tmax"]
weather
```

<div class="output execute_result" data-execution_count="14">

``` 
                station                              name  prcp  snow  snwd  \
DATE                                                                          
1970-01-01  USW00094789  JFK INTERNATIONAL AIRPORT, NY US  0.00   0.0   0.0   
1970-02-01  USW00094789  JFK INTERNATIONAL AIRPORT, NY US  0.00   0.0   0.0   
1970-03-01  USW00094789  JFK INTERNATIONAL AIRPORT, NY US  0.02   0.0   0.0   
1970-04-01  USW00094789  JFK INTERNATIONAL AIRPORT, NY US  0.00   0.0   0.0   
1970-05-01  USW00094789  JFK INTERNATIONAL AIRPORT, NY US  0.00   0.0   0.0   
...                 ...                               ...   ...   ...   ...   
2022-04-12  USW00094789  JFK INTERNATIONAL AIRPORT, NY US  0.00   0.0   0.0   
2022-05-12  USW00094789  JFK INTERNATIONAL AIRPORT, NY US  0.00   0.0   0.0   
2022-06-12  USW00094789  JFK INTERNATIONAL AIRPORT, NY US  0.43   0.0   0.0   
2022-07-12  USW00094789  JFK INTERNATIONAL AIRPORT, NY US  0.31   0.0   0.0   
2022-08-12  USW00094789  JFK INTERNATIONAL AIRPORT, NY US  0.00   0.0   0.0   

            tmax  tmin  target  
DATE                            
1970-01-01    28    22    31.0  
1970-02-01    31    22    38.0  
1970-03-01    38    25    31.0  
1970-04-01    31    23    35.0  
1970-05-01    35    21    36.0  
...          ...   ...     ...  
2022-04-12    47    37    47.0  
2022-05-12    47    30    57.0  
2022-06-12    57    36    57.0  
2022-07-12    57    54    55.0  
2022-08-12    55    42     NaN  

[19335 rows x 8 columns]
```

</div>

</div>

<div class="cell markdown" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2022-12-20T10:45:36.338685&quot;,&quot;end_time&quot;:&quot;2022-12-20T10:45:36.355138&quot;,&quot;duration&quot;:1.6453e-2}" data-tags="[]">

We can see we now have a Target column and it is tomorrow's `T-Max` so
for January 1st 1970 the target is 31 and that is tomorrow's maximum
temperature January 2nd which is 31. So you can see this helps us use
all of this data to predict tomorrow's temperature which is the target.

Now you may notice a little issue with the last value here it's missing
because we don't have data for December 8th. It's actually missing right
we don't have a value to pull back to know tomorrow's temperature so if
we made a prediction for this row we would actually be predicting
data that we don't have we'd be predicting the future so typically you
want to handle this in a different way but just to make things easier
and to make it easy to get future predictions what I'm going to do is
I'm going to use `ffill` again. We're going to pull the temperature the
target from the last row forward by using `ffill`.

</div>

<div class="cell code" data-execution_count="15" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-12-20T10:45:36.418163Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-12-20T10:45:36.391126Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-12-20T10:45:36.419662Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-12-20T10:45:36.390646Z&quot;}" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2022-12-20T10:45:36.372164&quot;,&quot;end_time&quot;:&quot;2022-12-20T10:45:36.421989&quot;,&quot;duration&quot;:4.9825e-2}" data-tags="[]">

``` python
weather = weather.ffill()
weather
```

<div class="output execute_result" data-execution_count="15">

``` 
                station                              name  prcp  snow  snwd  \
DATE                                                                          
1970-01-01  USW00094789  JFK INTERNATIONAL AIRPORT, NY US  0.00   0.0   0.0   
1970-02-01  USW00094789  JFK INTERNATIONAL AIRPORT, NY US  0.00   0.0   0.0   
1970-03-01  USW00094789  JFK INTERNATIONAL AIRPORT, NY US  0.02   0.0   0.0   
1970-04-01  USW00094789  JFK INTERNATIONAL AIRPORT, NY US  0.00   0.0   0.0   
1970-05-01  USW00094789  JFK INTERNATIONAL AIRPORT, NY US  0.00   0.0   0.0   
...                 ...                               ...   ...   ...   ...   
2022-04-12  USW00094789  JFK INTERNATIONAL AIRPORT, NY US  0.00   0.0   0.0   
2022-05-12  USW00094789  JFK INTERNATIONAL AIRPORT, NY US  0.00   0.0   0.0   
2022-06-12  USW00094789  JFK INTERNATIONAL AIRPORT, NY US  0.43   0.0   0.0   
2022-07-12  USW00094789  JFK INTERNATIONAL AIRPORT, NY US  0.31   0.0   0.0   
2022-08-12  USW00094789  JFK INTERNATIONAL AIRPORT, NY US  0.00   0.0   0.0   

            tmax  tmin  target  
DATE                            
1970-01-01    28    22    31.0  
1970-02-01    31    22    38.0  
1970-03-01    38    25    31.0  
1970-04-01    31    23    35.0  
1970-05-01    35    21    36.0  
...          ...   ...     ...  
2022-04-12    47    37    47.0  
2022-05-12    47    30    57.0  
2022-06-12    57    36    57.0  
2022-07-12    57    54    55.0  
2022-08-12    55    42    55.0  

[19335 rows x 8 columns]
```

</div>

</div>

<div class="cell markdown" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2022-12-20T10:45:36.439427&quot;,&quot;end_time&quot;:&quot;2022-12-20T10:45:36.457396&quot;,&quot;duration&quot;:1.7969e-2}" data-tags="[]">

We can see that we filled in yesterday's value to be the target here
it's going to cause a very very tiny issue that's not going to make a
difference to us right this row has a Target that that's actually not
correct but because we have 20 000 rows of data this one robe having an
incorrect Target is not going to make a huge difference and it'll make
it a little bit easier for us to make future predictions so that's why
I'm using fill there even though it's not technically correct it's not
really going to cause a problem.

</div>

<div class="cell markdown" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2022-12-20T10:45:36.474936&quot;,&quot;end_time&quot;:&quot;2022-12-20T10:45:36.491820&quot;,&quot;duration&quot;:1.6884e-2}" data-tags="[]">

We're going to apply a ridge regression model. You can check for
collinearity by using the `.corr()` method and this will actually
find the correlations between the various columns.

</div>

<div class="cell code" data-execution_count="16" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-12-20T10:45:37.850454Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-12-20T10:45:36.532487Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-12-20T10:45:37.851841Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-12-20T10:45:36.531246Z&quot;}" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2022-12-20T10:45:36.510304&quot;,&quot;end_time&quot;:&quot;2022-12-20T10:45:37.855078&quot;,&quot;duration&quot;:1.344774}" data-tags="[]">

``` python
from sklearn.linear_model import Ridge
```

</div>

<div class="cell code" data-execution_count="17" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-12-20T10:45:37.911722Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-12-20T10:45:37.892914Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-12-20T10:45:37.912786Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-12-20T10:45:37.892457Z&quot;}" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2022-12-20T10:45:37.873434&quot;,&quot;end_time&quot;:&quot;2022-12-20T10:45:37.915277&quot;,&quot;duration&quot;:4.1843e-2}" data-tags="[]">

``` python
weather.corr()
```

<div class="output execute_result" data-execution_count="17">

``` 
            prcp      snow      snwd      tmax      tmin    target
prcp    1.000000  0.151036  0.001223 -0.007102  0.052368 -0.003275
snow    0.151036  1.000000  0.232600 -0.174937 -0.159138 -0.172596
snwd    0.001223  0.232600  1.000000 -0.259645 -0.256740 -0.240846
tmax   -0.007102 -0.174937 -0.259645  1.000000  0.955414  0.915158
tmin    0.052368 -0.159138 -0.256740  0.955414  1.000000  0.915302
target -0.003275 -0.172596 -0.240846  0.915158  0.915302  1.000000
```

</div>

</div>

<div class="cell markdown" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2022-12-20T10:45:37.932745&quot;,&quot;end_time&quot;:&quot;2022-12-20T10:45:37.949552&quot;,&quot;duration&quot;:1.6807e-2}" data-tags="[]">

We can see that precipitation is pretty uncorrelated from most of the
columns and it's slightly correlated to snow. `tmax` and `tmin`
are pretty correlated which makes sense and the target is pretty
correlated with `tmax` and `tmin` which also makes sense that tomorrow's
temperature is pretty correlated to today's temperature.

</div>

<div class="cell markdown" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2022-12-20T10:45:37.966908&quot;,&quot;end_time&quot;:&quot;2022-12-20T10:45:37.983962&quot;,&quot;duration&quot;:1.7054e-2}" data-tags="[]">

We'll apply a [ridge regression machine learning
model](https://en.wikipedia.org/wiki/Ridge_regression#:~:text=Ridge%20regression%20is%20a%20method%20of%20estimating%20the,uses%20in%20fields%20including%20econometrics%2C%20chemistry%2C%20and%20engineering.)
which works very similarly to linear regression except it penalizes
coefficients to account for multi-collinearity. Ridge regression to
some extent helps adjust for collinearity.

Import Ridge Regression from scikit-learn and initialize the model. The
alpha parameter controls how much the coefficients are shrunk to account
for collinearity. It's worth experimenting by setting alpha to different
values. 0.1 is a good default value so we initialize our Ridge 
regression model and then the ridge regression model can be applied.

</div>

<div class="cell code" data-execution_count="18" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-12-20T10:45:38.025392Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-12-20T10:45:38.022041Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-12-20T10:45:38.026358Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-12-20T10:45:38.021225Z&quot;}" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2022-12-20T10:45:38.001287&quot;,&quot;end_time&quot;:&quot;2022-12-20T10:45:38.028846&quot;,&quot;duration&quot;:2.7559e-2}" data-tags="[]">

``` python
from sklearn.linear_model import Ridge

rr = Ridge(alpha=.1)
```

</div>

<div class="cell markdown" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2022-12-20T10:45:38.046869&quot;,&quot;end_time&quot;:&quot;2022-12-20T10:45:38.063780&quot;,&quot;duration&quot;:1.6911e-2}" data-tags="[]">

We need to create a list of predictor columns to predict our target and
to get that we're going to again index our list of columns.

</div>

<div class="cell code" data-execution_count="19" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-12-20T10:45:38.107862Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-12-20T10:45:38.101926Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-12-20T10:45:38.109412Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-12-20T10:45:38.101125Z&quot;}" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2022-12-20T10:45:38.081623&quot;,&quot;end_time&quot;:&quot;2022-12-20T10:45:38.111954&quot;,&quot;duration&quot;:3.0331e-2}" data-tags="[]">

``` python
predictors = weather.columns[~weather.columns.isin(["target", "name", "station"])]
predictors
```

<div class="output execute_result" data-execution_count="19">

    Index(['prcp', 'snow', 'snwd', 'tmax', 'tmin'], dtype='object')

</div>

</div>

<div class="cell markdown" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2022-12-20T10:45:38.129728&quot;,&quot;end_time&quot;:&quot;2022-12-20T10:45:38.146529&quot;,&quot;duration&quot;:1.6801e-2}" data-tags="[]">

So we can take a look at predictors and we can see this is our list of
predictors. We also have the time series data where January 1st is
linked to our data from January 2nd.

</div>

<div class="cell markdown" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2022-12-20T10:45:38.163527&quot;,&quot;end_time&quot;:&quot;2022-12-20T10:45:38.180244&quot;,&quot;duration&quot;:1.6717e-2}" data-tags="[]">

Typically when you try to estimate the error of a machine learning model
you can use cross-validation except with time series data. With time
series data we need to be really careful not to use future data to
predict the past. So we need to actually use a special technique called
back testing or time series cross-validation to account for the time
series nature of the data.

So we're going to write a function called `backtest` and this
function is going to take in our weather data frame, our ridge
regression model and our list of predictors. It's also going to define a
start parameter, `start=3650`, so this is how much data we want to take
- 10 years of data before we start making predictions and then
we're going to define a step, `step=90`, whichs means that every 90
days we'll create a set of predictions and then move on to the next 90
days and then the next 90 days so this is going to generate predictions
for our entire set of data except the first 10 years so we're going to
have predictions from 1980 all the way through 2022 and the
predictions will be will respect the order of the data.

</div>

<div class="cell markdown" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2022-12-20T10:45:38.197264&quot;,&quot;end_time&quot;:&quot;2022-12-20T10:45:38.214217&quot;,&quot;duration&quot;:1.6953e-2}" data-tags="[]">

So first thing we're going to create a list called `all_predictions`
and each element in this list is going to be a data frame that has
predictions for 90 days. Then we're going to write a `for` loop where we
say that for `i` in range and we're going to start with our start
parameter which is 3650 we're going to go up to `weather.shape[0]` which
is the end of our data set and then we are going to advance 90 each
time then work at each iteration. We're going to create a training set
so the training set is the data we use to train our machine
learning model and this is going to be all of the rows in our data up
to row `i` and then we're going to create a test set which is going to
be `i` up to `i+step`. So this is going to take all of the data that
comes before the current row to use as our training data and then this
is going to take the next 90 days to make predictions on then we're
going to go ahead and fit our model scikit-learn makes this really easy
we just call `model.fit` and then we pass in our predictors so our
predictors are what we're using to actually make our judgments and then
we're going to pass in our `target` and our `target` is what we're
trying to predict.

</div>

<div class="cell code" data-execution_count="20" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-12-20T10:45:38.260592Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-12-20T10:45:38.250814Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-12-20T10:45:38.261534Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-12-20T10:45:38.250307Z&quot;}" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2022-12-20T10:45:38.231386&quot;,&quot;end_time&quot;:&quot;2022-12-20T10:45:38.263642&quot;,&quot;duration&quot;:3.2256e-2}" data-tags="[]">

``` python
def backtest(weather, model, predictors, start=3650, step=90):
    all_predictions = []
    
    for i in range(start, weather.shape[0], step):    #weather.shape[0] represents end of the dataset
        train = weather.iloc[:i,:]             #create an iteration at each 
                                               #training set for all of the rows upto row i
        test = weather.iloc[i:(i+step),:]      #test set takes the next 90 days
        
        model.fit(train[predictors], train["target"])    #use the scikit learn to fit the
                                                         #ridge regression to the data
        preds = model.predict(test[predictors])          #generate the predictions using scikit learn
        preds = pd.Series(preds, index=test.index)       #convert the predictions into a pandas series
        combined = pd.concat([test["target"], preds], axis=1)   #concatenate the real test data with the predictions data
                                                                #axis=1 treats each of the targets as a separate column in a single df
        combined.columns = ["actual", "prediction"]             #name the columns
        combined["diff"] = (combined["prediction"] - combined["actual"]).abs()    #take the absolute value of the difference
        
        all_predictions.append(combined)
    return pd.concat(all_predictions)
```

</div>

<div class="cell markdown" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2022-12-20T10:45:38.280761&quot;,&quot;end_time&quot;:&quot;2022-12-20T10:45:38.297451&quot;,&quot;duration&quot;:1.669e-2}" data-tags="[]">

Make prediction by calling the function.

</div>

<div class="cell code" data-execution_count="21" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-12-20T10:45:39.883751Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-12-20T10:45:38.334267Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-12-20T10:45:39.886074Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-12-20T10:45:38.332995Z&quot;}" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2022-12-20T10:45:38.314464&quot;,&quot;end_time&quot;:&quot;2022-12-20T10:45:39.889011&quot;,&quot;duration&quot;:1.574547}" data-tags="[]">

``` python
predictions = backtest(weather, rr, predictors)
predictions
```

<div class="output execute_result" data-execution_count="21">

``` 
            actual  prediction       diff
DATE                                     
1979-12-30    43.0   50.229324   7.229324
1979-12-31    42.0   43.673798   1.673798
1980-01-01    41.0   41.579150   0.579150
1980-02-01    36.0   43.961887   7.961887
1980-03-01    30.0   40.204726  10.204726
...            ...         ...        ...
2022-04-12    47.0   49.901241   2.901241
2022-05-12    57.0   46.282630  10.717370
2022-06-12    57.0   53.330679   3.669321
2022-07-12    55.0   62.783822   7.783822
2022-08-12    55.0   56.067751   1.067751

[15685 rows x 3 columns]
```

</div>

</div>

<div class="cell markdown" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2022-12-20T10:45:39.907748&quot;,&quot;end_time&quot;:&quot;2022-12-20T10:45:39.927922&quot;,&quot;duration&quot;:2.0174e-2}" data-tags="[]">

We can take a look at our predictions and we can see that we skipped the
first 10 years because we use 10 years of data to make our first set of
predictions but we have predictions from the end of 1979 onwards through
to 2022.

In order to figure out how good our predictions were, generate an
accuracy metric. The metric we're going to use is mean absolute error.
It's basically just taking this diff column and finding the average.

</div>

<div class="cell code" data-execution_count="22" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-12-20T10:45:39.974692Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-12-20T10:45:39.968780Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-12-20T10:45:39.975444Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-12-20T10:45:39.967969Z&quot;}" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2022-12-20T10:45:39.948071&quot;,&quot;end_time&quot;:&quot;2022-12-20T10:45:39.977632&quot;,&quot;duration&quot;:2.9561e-2}" data-tags="[]">

``` python
from sklearn.metrics import mean_absolute_error, mean_squared_error

mean_absolute_error(predictions["actual"], predictions["prediction"])
```

<div class="output execute_result" data-execution_count="22">

    5.140877250169638

</div>

</div>

<div class="cell markdown" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2022-12-20T10:45:39.995709&quot;,&quot;end_time&quot;:&quot;2022-12-20T10:45:40.013294&quot;,&quot;duration&quot;:1.7585e-2}" data-tags="[]">

It means on average we were five degrees off from the correct
temperature on average it means about half the time we were further off
half the time we were we were closer so not great accuracy I mean
five degrees Fahrenheit isn't a huge error but we can do better.

</div>

<div class="cell code" data-execution_count="23" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-12-20T10:45:40.056583Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-12-20T10:45:40.051532Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-12-20T10:45:40.057397Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-12-20T10:45:40.050736Z&quot;}" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2022-12-20T10:45:40.031177&quot;,&quot;end_time&quot;:&quot;2022-12-20T10:45:40.059646&quot;,&quot;duration&quot;:2.8469e-2}" data-tags="[]">

``` python
predictions["diff"].mean()
```

<div class="output execute_result" data-execution_count="23">

    5.140877250169638

</div>

</div>

<div class="cell markdown" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2022-12-20T10:45:40.077782&quot;,&quot;end_time&quot;:&quot;2022-12-20T10:45:40.096873&quot;,&quot;duration&quot;:1.9091e-2}" data-tags="[]">

The way we're going to improve our accuracy is by calculating the
average temperature and precipitation in the past few days so the
past three days and the past 14 days and looking at how the current day
compares to those days.

</div>

<div class="cell code" data-execution_count="24" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-12-20T10:45:40.187928Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-12-20T10:45:40.135384Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-12-20T10:45:40.189321Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-12-20T10:45:40.134442Z&quot;}" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2022-12-20T10:45:40.114855&quot;,&quot;end_time&quot;:&quot;2022-12-20T10:45:40.192496&quot;,&quot;duration&quot;:7.7641e-2}" data-tags="[]">

``` python
def pct_diff(old, new):     #define a function to calculate percent_difference
    return (new - old) / old

def compute_rolling(weather, horizon, col):     #calculate rolling averages for the past few periods
                                                #horizon is the no. of days you want to calculate rolling average for
    label = f"rolling_{horizon}_{col}"          #label for the new column
    weather[label] = weather[col].rolling(horizon).mean()    #create a new column that contains the 14 day rolling average
    weather[f"{label}_pct"] = pct_diff(weather[label], weather[col])    #find the difference between the current day 
                                                                        #and the rolling
    return weather
    
rolling_horizons = [3, 14]     #run it for a 3-day horizon and 14-day horizon
for horizon in rolling_horizons:
    for col in ["tmax", "tmin", "prcp"]:
        weather = compute_rolling(weather, horizon, col)
        
weather
```

<div class="output execute_result" data-execution_count="24">

``` 
                station                              name  prcp  snow  snwd  \
DATE                                                                          
1970-01-01  USW00094789  JFK INTERNATIONAL AIRPORT, NY US  0.00   0.0   0.0   
1970-02-01  USW00094789  JFK INTERNATIONAL AIRPORT, NY US  0.00   0.0   0.0   
1970-03-01  USW00094789  JFK INTERNATIONAL AIRPORT, NY US  0.02   0.0   0.0   
1970-04-01  USW00094789  JFK INTERNATIONAL AIRPORT, NY US  0.00   0.0   0.0   
1970-05-01  USW00094789  JFK INTERNATIONAL AIRPORT, NY US  0.00   0.0   0.0   
...                 ...                               ...   ...   ...   ...   
2022-04-12  USW00094789  JFK INTERNATIONAL AIRPORT, NY US  0.00   0.0   0.0   
2022-05-12  USW00094789  JFK INTERNATIONAL AIRPORT, NY US  0.00   0.0   0.0   
2022-06-12  USW00094789  JFK INTERNATIONAL AIRPORT, NY US  0.43   0.0   0.0   
2022-07-12  USW00094789  JFK INTERNATIONAL AIRPORT, NY US  0.31   0.0   0.0   
2022-08-12  USW00094789  JFK INTERNATIONAL AIRPORT, NY US  0.00   0.0   0.0   

            tmax  tmin  target  rolling_3_tmax  rolling_3_tmax_pct  \
DATE                                                                 
1970-01-01    28    22    31.0             NaN                 NaN   
1970-02-01    31    22    38.0             NaN                 NaN   
1970-03-01    38    25    31.0       32.333333            0.175258   
1970-04-01    31    23    35.0       33.333333           -0.070000   
1970-05-01    35    21    36.0       34.666667            0.009615   
...          ...   ...     ...             ...                 ...   
2022-04-12    47    37    47.0       49.666667           -0.053691   
2022-05-12    47    30    57.0       50.333333           -0.066225   
2022-06-12    57    36    57.0       50.333333            0.132450   
2022-07-12    57    54    55.0       53.666667            0.062112   
2022-08-12    55    42    55.0       56.333333           -0.023669   

            rolling_3_tmin  rolling_3_tmin_pct  rolling_3_prcp  \
DATE                                                             
1970-01-01             NaN                 NaN             NaN   
1970-02-01             NaN                 NaN             NaN   
1970-03-01       23.000000            0.086957        0.006667   
1970-04-01       23.333333           -0.014286        0.006667   
1970-05-01       23.000000           -0.086957        0.006667   
...                    ...                 ...             ...   
2022-04-12       37.333333           -0.008929        0.076667   
2022-05-12       37.000000           -0.189189        0.076667   
2022-06-12       34.333333            0.048544        0.143333   
2022-07-12       40.000000            0.350000        0.246667   
2022-08-12       44.000000           -0.045455        0.246667   

            rolling_3_prcp_pct  rolling_14_tmax  rolling_14_tmax_pct  \
DATE                                                                   
1970-01-01                 NaN              NaN                  NaN   
1970-02-01                 NaN              NaN                  NaN   
1970-03-01            2.000000              NaN                  NaN   
1970-04-01           -1.000000              NaN                  NaN   
1970-05-01           -1.000000              NaN                  NaN   
...                        ...              ...                  ...   
2022-04-12           -1.000000        51.785714            -0.092414   
2022-05-12           -1.000000        52.142857            -0.098630   
2022-06-12            2.000000        52.642857             0.082768   
2022-07-12            0.256757        52.642857             0.082768   
2022-08-12           -1.000000        52.785714             0.041949   

            rolling_14_tmin  rolling_14_tmin_pct  rolling_14_prcp  \
DATE                                                                
1970-01-01              NaN                  NaN              NaN   
1970-02-01              NaN                  NaN              NaN   
1970-03-01              NaN                  NaN              NaN   
1970-04-01              NaN                  NaN              NaN   
1970-05-01              NaN                  NaN              NaN   
...                     ...                  ...              ...   
2022-04-12        37.000000             0.000000         0.082143   
2022-05-12        37.142857            -0.192308         0.082143   
2022-06-12        37.071429            -0.028902         0.112857   
2022-07-12        38.500000             0.402597         0.135000   
2022-08-12        39.142857             0.072993         0.135000   

            rolling_14_prcp_pct  
DATE                             
1970-01-01                  NaN  
1970-02-01                  NaN  
1970-03-01                  NaN  
1970-04-01                  NaN  
1970-05-01                  NaN  
...                         ...  
2022-04-12            -1.000000  
2022-05-12            -1.000000  
2022-06-12             2.810127  
2022-07-12             1.296296  
2022-08-12            -1.000000  

[19335 rows x 20 columns]
```

</div>

</div>

<div class="cell markdown" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2022-12-20T10:45:40.211228&quot;,&quot;end_time&quot;:&quot;2022-12-20T10:45:40.230614&quot;,&quot;duration&quot;:1.9386e-2}" data-tags="[]">

You can see there's a few missing values so the reason for this is if
we're finding a 14-day rolling average for these dates we don't have 14
days of historical data to compute a rolling average. Our data started
on January 1st 1970 so there aren't 14 days previous to that that we
have data from so pandas has has basically labeled all these rows
missing.

On looking at the weather dataframe we observe that there are still a
couple of missing values here and that is in the percentage column so
this happens when we're basically dividing by either dividing by zero or
dividing zero. `weather.fillna(0)` means find any missing values and
fill them in with zero.

</div>

<div class="cell code" data-execution_count="25" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-12-20T10:45:40.287963Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-12-20T10:45:40.270215Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-12-20T10:45:40.289442Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-12-20T10:45:40.269803Z&quot;}" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2022-12-20T10:45:40.249859&quot;,&quot;end_time&quot;:&quot;2022-12-20T10:45:40.292617&quot;,&quot;duration&quot;:4.2758e-2}" data-tags="[]">

``` python
weather = weather.iloc[14:,:]
weather = weather.fillna(0)
```

</div>

<div class="cell markdown" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2022-12-20T10:45:40.311284&quot;,&quot;end_time&quot;:&quot;2022-12-20T10:45:40.329444&quot;,&quot;duration&quot;:1.816e-2}" data-tags="[]">

All right and now what we can do is we can add a couple more predictors
so we'll first write a function called `expand_mean` and this is going
to take our data frame as input return the mean of all of those rows
together.

</div>

<div class="cell code" data-execution_count="26" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-12-20T10:45:40.707644Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-12-20T10:45:40.369030Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-12-20T10:45:40.708865Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-12-20T10:45:40.368577Z&quot;}" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2022-12-20T10:45:40.348312&quot;,&quot;end_time&quot;:&quot;2022-12-20T10:45:40.711809&quot;,&quot;duration&quot;:0.363497}" data-tags="[]">

``` python
def expand_mean(df):
    return df.expanding(1).mean()

for col in ["tmax", "tmin", "prcp"]:
    weather[f"month_avg_{col}"] = weather[col].groupby(weather.index.month, group_keys=False).apply(expand_mean)     #group it by month
    weather[f"day_avg_{col}"] = weather[col].groupby(weather.index.day_of_year, group_keys=False).apply(expand_mean) #group it by day
```

</div>

<div class="cell markdown" data-_kg_hide-output="true" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2022-12-20T10:45:40.730070&quot;,&quot;end_time&quot;:&quot;2022-12-20T10:45:40.747793&quot;,&quot;duration&quot;:1.7723e-2}" data-tags="[]">

`group_keys` equals false this just tells pandas to make the output
clean and not include another level to the index. The function
`expand_mean` groups the data by the month.

</div>

<div class="cell code" data-execution_count="27" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-12-20T10:45:40.827191Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-12-20T10:45:40.786222Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-12-20T10:45:40.828360Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-12-20T10:45:40.785388Z&quot;}" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2022-12-20T10:45:40.765848&quot;,&quot;end_time&quot;:&quot;2022-12-20T10:45:40.831182&quot;,&quot;duration&quot;:6.5334e-2}" data-tags="[]">

``` python
weather
```

<div class="output execute_result" data-execution_count="27">

``` 
                station                              name  prcp  snow  snwd  \
DATE                                                                          
1970-01-15  USW00094789  JFK INTERNATIONAL AIRPORT, NY US  0.00   0.0   0.0   
1970-01-16  USW00094789  JFK INTERNATIONAL AIRPORT, NY US  0.00   0.0   0.0   
1970-01-17  USW00094789  JFK INTERNATIONAL AIRPORT, NY US  0.02   0.0   0.0   
1970-01-18  USW00094789  JFK INTERNATIONAL AIRPORT, NY US  0.10   0.0   0.0   
1970-01-19  USW00094789  JFK INTERNATIONAL AIRPORT, NY US  0.00   0.0   0.0   
...                 ...                               ...   ...   ...   ...   
2022-04-12  USW00094789  JFK INTERNATIONAL AIRPORT, NY US  0.00   0.0   0.0   
2022-05-12  USW00094789  JFK INTERNATIONAL AIRPORT, NY US  0.00   0.0   0.0   
2022-06-12  USW00094789  JFK INTERNATIONAL AIRPORT, NY US  0.43   0.0   0.0   
2022-07-12  USW00094789  JFK INTERNATIONAL AIRPORT, NY US  0.31   0.0   0.0   
2022-08-12  USW00094789  JFK INTERNATIONAL AIRPORT, NY US  0.00   0.0   0.0   

            tmax  tmin  target  rolling_3_tmax  rolling_3_tmax_pct  ...  \
DATE                                                                ...   
1970-01-15    29    13    36.0       29.666667           -0.022472  ...   
1970-01-16    36    21    43.0       30.333333            0.186813  ...   
1970-01-17    43    30    42.0       36.000000            0.194444  ...   
1970-01-18    42    25    25.0       40.333333            0.041322  ...   
1970-01-19    25    16    24.0       36.666667           -0.318182  ...   
...          ...   ...     ...             ...                 ...  ...   
2022-04-12    47    37    47.0       49.666667           -0.053691  ...   
2022-05-12    47    30    57.0       50.333333           -0.066225  ...   
2022-06-12    57    36    57.0       50.333333            0.132450  ...   
2022-07-12    57    54    55.0       53.666667            0.062112  ...   
2022-08-12    55    42    55.0       56.333333           -0.023669  ...   

            rolling_14_tmin  rolling_14_tmin_pct  rolling_14_prcp  \
DATE                                                                
1970-01-15        18.857143            -0.310606         0.022857   
1970-01-16        18.785714             0.117871         0.022857   
1970-01-17        19.142857             0.567164         0.022857   
1970-01-18        19.285714             0.296296         0.030000   
1970-01-19        18.928571            -0.154717         0.030000   
...                     ...                  ...              ...   
2022-04-12        37.000000             0.000000         0.082143   
2022-05-12        37.142857            -0.192308         0.082143   
2022-06-12        37.071429            -0.028902         0.112857   
2022-07-12        38.500000             0.402597         0.135000   
2022-08-12        39.142857             0.072993         0.135000   

            rolling_14_prcp_pct  month_avg_tmax  day_avg_tmax  month_avg_tmin  \
DATE                                                                            
1970-01-15            -1.000000       29.000000     29.000000       13.000000   
1970-01-16            -1.000000       32.500000     36.000000       17.000000   
1970-01-17            -0.125000       36.000000     43.000000       21.333333   
1970-01-18             2.333333       37.500000     42.000000       22.250000   
1970-01-19            -1.000000       35.000000     25.000000       21.000000   
...                         ...             ...           ...             ...   
2022-04-12            -1.000000       61.531781     51.075472       46.358087   
2022-05-12            -1.000000       67.052375     49.339623       52.104141   
2022-06-12             2.810127       72.006293     50.509434       57.427313   
2022-07-12             1.296296       75.300244     49.490566       60.901340   
2022-08-12            -1.000000       73.801462     46.754717       59.528015   

            day_avg_tmin  month_avg_prcp  day_avg_prcp  
DATE                                                    
1970-01-15     13.000000        0.000000      0.000000  
1970-01-16     21.000000        0.000000      0.000000  
1970-01-17     30.000000        0.006667      0.020000  
1970-01-18     25.000000        0.030000      0.100000  
1970-01-19     16.000000        0.024000      0.000000  
...                  ...             ...           ...  
2022-04-12     37.000000        0.120441      0.099245  
2022-05-12     38.113208        0.121066      0.146038  
2022-06-12     38.094340        0.111775      0.111321  
2022-07-12     36.169811        0.122856      0.059434  
2022-08-12     34.754717        0.123167      0.139057  

[19321 rows x 26 columns]
```

</div>

</div>

<div class="cell markdown" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2022-12-20T10:45:40.850588&quot;,&quot;end_time&quot;:&quot;2022-12-20T10:45:40.868948&quot;,&quot;duration&quot;:1.836e-2}" data-tags="[]">

We have our new set of predictors. We've added a lot more columns to our
data frame and we want those to be picked up by our list of predictors
so let's  go ahead and take a look.

</div>

<div class="cell code" data-execution_count="28" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-12-20T10:45:40.916931Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-12-20T10:45:40.908575Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-12-20T10:45:40.918085Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-12-20T10:45:40.908155Z&quot;}" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2022-12-20T10:45:40.888245&quot;,&quot;end_time&quot;:&quot;2022-12-20T10:45:40.920416&quot;,&quot;duration&quot;:3.2171e-2}" data-tags="[]">

``` python
predictors = weather.columns[~weather.columns.isin(["target", "name", "station"])]
predictors
```

<div class="output execute_result" data-execution_count="28">

    Index(['prcp', 'snow', 'snwd', 'tmax', 'tmin', 'rolling_3_tmax',
           'rolling_3_tmax_pct', 'rolling_3_tmin', 'rolling_3_tmin_pct',
           'rolling_3_prcp', 'rolling_3_prcp_pct', 'rolling_14_tmax',
           'rolling_14_tmax_pct', 'rolling_14_tmin', 'rolling_14_tmin_pct',
           'rolling_14_prcp', 'rolling_14_prcp_pct', 'month_avg_tmax',
           'day_avg_tmax', 'month_avg_tmin', 'day_avg_tmin', 'month_avg_prcp',
           'day_avg_prcp'],
          dtype='object')

</div>

</div>

<div class="cell markdown" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2022-12-20T10:45:40.939128&quot;,&quot;end_time&quot;:&quot;2022-12-20T10:45:40.957355&quot;,&quot;duration&quot;:1.8227e-2}" data-tags="[]">

We can see these are our new predictors that we're going to use. We can
actually just call the `backtest` function again and just pass in our
new weather data frame and our new predictors and it'll give us updated
predictions.

</div>

<div class="cell code" data-execution_count="29" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-12-20T10:45:46.191739Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-12-20T10:45:40.997001Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-12-20T10:45:46.193330Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-12-20T10:45:40.996493Z&quot;}" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2022-12-20T10:45:40.976006&quot;,&quot;end_time&quot;:&quot;2022-12-20T10:45:46.197547&quot;,&quot;duration&quot;:5.221541}" data-tags="[]">

``` python
predictions = backtest(weather, rr, predictors)
mean_absolute_error(predictions["actual"], predictions["prediction"])
```

<div class="output execute_result" data-execution_count="29">

    4.854152302923006

</div>

</div>

<div class="cell markdown" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2022-12-20T10:45:46.243546&quot;,&quot;end_time&quot;:&quot;2022-12-20T10:45:46.262314&quot;,&quot;duration&quot;:1.8768e-2}" data-tags="[]">

Then we can again just do mean absolute error to find our error. so it
is a good bit lower now which is great

</div>

<div class="cell code" data-execution_count="30" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-12-20T10:45:46.308601Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-12-20T10:45:46.302214Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-12-20T10:45:46.309474Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-12-20T10:45:46.301765Z&quot;}" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2022-12-20T10:45:46.281382&quot;,&quot;end_time&quot;:&quot;2022-12-20T10:45:46.311812&quot;,&quot;duration&quot;:3.043e-2}" data-tags="[]">

``` python
mean_squared_error(predictions["actual"], predictions["prediction"])
```

<div class="output execute_result" data-execution_count="30">

    38.60237089906105

</div>

</div>

<div class="cell markdown" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2022-12-20T10:45:46.331415&quot;,&quot;end_time&quot;:&quot;2022-12-20T10:45:46.351023&quot;,&quot;duration&quot;:1.9608e-2}" data-tags="[]">

We can take a look just at our predictions data frame and we can do sort
values which will sort our data frame by a single column in this case
we're going to sort it  by the difference between what we predicted
and the actual temperature. Sort them in descending order so we can see
the days on which we had our biggest errors these are typically days
where the temperature the day before was a lot lower and the temperature
the day after was a lot lower so they they're kind of anomalous days.

</div>

<div class="cell code" data-execution_count="31" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-12-20T10:45:46.409589Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-12-20T10:45:46.391986Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-12-20T10:45:46.410746Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-12-20T10:45:46.391446Z&quot;}" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2022-12-20T10:45:46.370408&quot;,&quot;end_time&quot;:&quot;2022-12-20T10:45:46.413302&quot;,&quot;duration&quot;:4.2894e-2}" data-tags="[]">

``` python
predictions.sort_values("diff", ascending=False)
```

<div class="output execute_result" data-execution_count="31">

``` 
            actual  prediction       diff
DATE                                     
1990-12-03    85.0   53.651017  31.348983
1998-03-26    80.0   51.347896  28.652104
2007-03-26    78.0   49.742780  28.257220
2003-04-15    86.0   57.987157  28.012843
1985-04-18    84.0   58.091652  25.908348
...            ...         ...        ...
1999-01-15    42.0   41.995849   0.004151
1982-12-18    37.0   37.002622   0.002622
2014-01-24    30.0   30.002059   0.002059
1995-03-09    80.0   80.000552   0.000552
1995-01-24    44.0   43.999697   0.000303

[15671 rows x 3 columns]
```

</div>

</div>

<div class="cell markdown" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2022-12-20T10:45:46.433002&quot;,&quot;end_time&quot;:&quot;2022-12-20T10:45:46.452161&quot;,&quot;duration&quot;:1.9159e-2}" data-tags="[]">

For example if we wanted to take a look at what happened on March 12
1990 we could say we want to take a look at all of the rows from March
7th to March 17 1990. and what this gives us is our rows immediately
before and immediately after the anomalous temperature so we can take a
look here and we can see 85 was was really seemed kind of random right
there are a lot of lower temperatures than 85 than a lot of lower
temperatures

</div>

<div class="cell code" data-execution_count="32" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-12-20T10:45:46.544436Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-12-20T10:45:46.494184Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-12-20T10:45:46.545813Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-12-20T10:45:46.493168Z&quot;}" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2022-12-20T10:45:46.471906&quot;,&quot;end_time&quot;:&quot;2022-12-20T10:45:46.548524&quot;,&quot;duration&quot;:7.6618e-2}" data-tags="[]">

``` python
weather.loc["1990-03-07": "1990-03-17"]
```

<div class="output execute_result" data-execution_count="32">

``` 
                station                              name  prcp  snow  snwd  \
DATE                                                                          
1990-03-13  USW00094789  JFK INTERNATIONAL AIRPORT, NY US  0.00   0.0   0.0   
1990-03-14  USW00094789  JFK INTERNATIONAL AIRPORT, NY US  0.00   0.0   0.0   
1990-03-15  USW00094789  JFK INTERNATIONAL AIRPORT, NY US  0.00   0.0   0.0   
1990-03-16  USW00094789  JFK INTERNATIONAL AIRPORT, NY US  0.00   0.0   0.0   
1990-03-17  USW00094789  JFK INTERNATIONAL AIRPORT, NY US  0.26   0.0   0.0   
1990-03-07  USW00094789  JFK INTERNATIONAL AIRPORT, NY US  0.00   0.0   0.0   
1990-03-08  USW00094789  JFK INTERNATIONAL AIRPORT, NY US  0.00   0.0   0.0   
1990-03-09  USW00094789  JFK INTERNATIONAL AIRPORT, NY US  0.00   0.0   0.0   
1990-03-10  USW00094789  JFK INTERNATIONAL AIRPORT, NY US  0.00   0.0   0.0   
1990-03-11  USW00094789  JFK INTERNATIONAL AIRPORT, NY US  0.00   0.0   0.0   
1990-03-12  USW00094789  JFK INTERNATIONAL AIRPORT, NY US  0.85   0.0   0.0   

            tmax  tmin  target  rolling_3_tmax  rolling_3_tmax_pct  ...  \
DATE                                                                ...   
1990-03-13    85    41    62.0       67.666667            0.256158  ...   
1990-03-14    62    46    55.0       68.666667           -0.097087  ...   
1990-03-15    55    43    62.0       67.333333           -0.183168  ...   
1990-03-16    62    48    61.0       59.666667            0.039106  ...   
1990-03-17    61    49    59.0       59.333333            0.028090  ...   
1990-03-07    85    63    90.0       79.666667            0.066946  ...   
1990-03-08    89    72    85.0       86.666667            0.026923  ...   
1990-03-09    77    63    74.0       79.333333           -0.029412  ...   
1990-03-10    69    50    73.0       69.333333           -0.004808  ...   
1990-03-11    75    53    70.0       69.666667            0.076555  ...   
1990-03-12    53    38    58.0       53.666667           -0.012422  ...   

            rolling_14_tmin  rolling_14_tmin_pct  rolling_14_prcp  \
DATE                                                                
1990-03-13        29.500000             0.389831         0.020000   
1990-03-14        30.857143             0.490741         0.020000   
1990-03-15        32.214286             0.334812         0.020000   
1990-03-16        33.428571             0.435897         0.020000   
1990-03-17        34.357143             0.426195         0.038571   
1990-03-07        66.285714            -0.049569         0.052857   
1990-03-08        70.071429             0.027523         0.197143   
1990-03-09        67.071429            -0.060703         0.229286   
1990-03-10        56.071429            -0.108280         0.057857   
1990-03-11        47.428571             0.117470         0.112143   
1990-03-12        39.500000            -0.037975         0.091429   

            rolling_14_prcp_pct  month_avg_tmax  day_avg_tmax  month_avg_tmin  \
DATE                                                                            
1990-03-13            -1.000000       54.794543     50.380952       40.662921   
1990-03-14            -1.000000       54.806090     50.190476       40.671474   
1990-03-15            -1.000000       54.806400     49.714286       40.675200   
1990-03-16            -1.000000       54.817891     50.095238       40.686901   
1990-03-17             5.740741       54.827751     48.095238       40.700159   
1990-03-07            -1.000000       54.793798     79.047619       40.668217   
1990-03-08            -1.000000       54.846749     83.904762       40.716718   
1990-03-09            -1.000000       54.880989     81.523810       40.751159   
1990-03-10            -1.000000       54.902778     70.619048       40.765432   
1990-03-11            -1.000000       54.933744     63.857143       40.784284   
1990-03-12             8.296875       54.930769     49.952381       40.780000   

            day_avg_tmin  month_avg_prcp  day_avg_prcp  
DATE                                                    
1990-03-13     35.380952        0.129470      0.076190  
1990-03-14     36.809524        0.129263      0.259524  
1990-03-15     35.761905        0.129056      0.066667  
1990-03-16     35.619048        0.128850      0.083810  
1990-03-17     34.619048        0.129059      0.079048  
1990-03-07     64.095238        0.130202      0.153810  
1990-03-08     70.380952        0.130000      0.080476  
1990-03-09     67.000000        0.129799      0.113810  
1990-03-10     56.095238        0.129599      0.178095  
1990-03-11     50.047619        0.129399      0.111905  
1990-03-12     35.761905        0.130508      0.097619  

[11 rows x 26 columns]
```

</div>

</div>

<div class="cell markdown" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2022-12-20T10:45:46.569660&quot;,&quot;end_time&quot;:&quot;2022-12-20T10:45:46.589395&quot;,&quot;duration&quot;:1.9735e-2}" data-tags="[]">

More atmospheric data about wind conditions, cloud cover and barometric
pressure can be used to predict accurately. We can make a plot of the
error bucket.

</div>

<div class="cell code" data-execution_count="33" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-12-20T10:45:46.794901Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-12-20T10:45:46.631367Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-12-20T10:45:46.796125Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-12-20T10:45:46.630907Z&quot;}" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2022-12-20T10:45:46.609371&quot;,&quot;end_time&quot;:&quot;2022-12-20T10:45:46.798633&quot;,&quot;duration&quot;:0.189262}" data-tags="[]">

``` python
predictions["diff"].round().value_counts().sort_index().plot()
```

<div class="output execute_result" data-execution_count="33">

    <AxesSubplot:>

</div>

<div class="output display_data">

![](e50449a1f39888ed3542c40d6a44f58a5b56b938.png)

</div>

</div>

<div class="cell markdown" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2022-12-20T10:45:46.819243&quot;,&quot;end_time&quot;:&quot;2022-12-20T10:45:46.839567&quot;,&quot;duration&quot;:2.0324e-2}" data-tags="[]">

We can see most of the time our error is pretty low but there's kind of
a long tail of error where some of the errors are actually very very
high and these are the things that make our mean absolute error a lot
higher.

</div>

<div class="cell markdown" data-papermill="{&quot;status&quot;:&quot;completed&quot;,&quot;exception&quot;:false,&quot;start_time&quot;:&quot;2022-12-20T10:45:46.860157&quot;,&quot;end_time&quot;:&quot;2022-12-20T10:45:46.880586&quot;,&quot;duration&quot;:2.0429e-2}" data-tags="[]">

We've built our model, improved our model and taken a look at a
few diagnostics to investigate what's happening with our model. To
continue improving the model you can do to improve accuracy is always to
add in more predictor columns. You can add in more columns like the
average monthly temperature, the average daily temperature or ratios
between the two or ratio between the current temperature or current
precipitation. You can also take a look at these types of rolling
averages and and see if you can compute either different Horizons or
different types of rolling predictors. 

Some column with null values were removed initially. We can actually use
some of these column, process them a little bit differently and
investigate if there is anything useful. The other thing you can do to
improve accuracy is actually to change the model - you could use xgboost
or random forest or a more complicated model that may perform better.
More complicated doesn't always mean that it will perform better but it
could so it's worth trying that to improve accuracy.

</div>
