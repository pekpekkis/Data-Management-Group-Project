# Data-Management-Group-Project
Group project on CO2 emissions and renewable energy

The repository includes:

  -Downloaded datasets used in the project
  
  -Different steps of the project (either as a python file or a jupyter notebook):
  
    -Data munging
    -Data visualization
    -Data modeling
    
The complete version is in the python file project_app.py. The app is a streamlit app which runs when going to the right directory in prompt and typing
"streamlit run project_app.py". The app includes every part of the project and begins with a page showcasing the carbon dioxide emissions and renewable energy usage by country.
The second page presents results of a log-linear regression, first with the whole dataset (with missing values removed) and second with outliers removed from the
data. Under the tables showcasing the regression results a few comments is made about the statistical significance and goodness-of-fit.

The packages that are needed to run the code:

    -Pandas
    -Geopandas
    -Matplotlib
    -Matplotlib.pyplot
    -Numpy
    -Folium
    -Streamlit
    -Seaborn
    -Statsmodels.api

Also, a written report as a pdf file can be found in the repository along with presentation slides. These include the motivation for the project and more ellaborative comments on the data and regression results. The report file is called Data management - Co2 emissions and renewable energy report.pdf and the presentation slides are The power point file named: Co2 emissions and renewable energy slide for exam.pptx .
