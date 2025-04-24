# Description:
    This project aims to predict flight delays using multiple datasets, including U.S. International Air Traffic data, Flight Delay and Cancellation data, Storm Events data, Airline Fleets data, and more. The main goal is to preprocess and analyze the data to uncover patterns and factors influencing flight delays. The project involves cleaning the data, performing exploratory data analysis (EDA), and preparing it for predictive modeling. The final model will help forecast flight delays based on historical and external factors such as weather and airfare trends.


# Team Members:
    1. Harsh Gupta (gupta.harsh@ufl.edu)
    2. Muthukumaran Ulaganathan (ulaganathan.m@ufl.edu)

# Datasets:
    1. [International Air Traffic data](https://www.kaggle.com/datasets/parulpandey/us-international-air-traffic-data?select=International_Report_Passengers.csv)
    2. [Airfare report](https://catalog.data.gov/dataset/consumer-airfare-report-table-5-detailed-fare-information-for-highest-and-lowest-fare-mark)
    3. [Flight delay and cancellation](https://www.kaggle.com/datasets/patrickzel/flight-delay-and-cancellation-dataset-2019-2023)
    4. [Storm Events data](https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/)
    5. [Airline Fleet data](https://www.kaggle.com/datasets/traceyvanp/airlinefleet)

# How to run?

- Clone the repo and cd into the working directory
- Run the individual *_eda notebooks to preprocess and EDA individual datasets
- Run the production merge notebook to join the datasets
- Run the production eda notebook to visualize the joined dataset

# Presentation

Presentation Video URL: https://drive.google.com/file/d/1cMTVvTEBltpU7w7afSgjFiwKaIIGqWcu/view?usp=share_link

Presentation PPT URL: https://docs.google.com/presentation/d/1IjdVWy3IgBzawYqLkyzi9TYJzfo3OrrTfs7Ctsm6AUg/edit?usp=share_link

Tool Demo URL: https://drive.google.com/file/d/1-lWoTndmbrZ44ByeRMWkrfJ7BG9ZO4OK/view?usp=share_link

## Team Contribution

### Why it was a 2-person project

This project required the integration of six diverse datasets, including international and domestic flight records, storm events, airfare reports, and airline fleet data. We engineered over 20 domain-specific features related to time, weather, and traffic. The team evaluated multiple machine learning models, including deep learning, and developed a full-featured Streamlit dashboard with both real-time prediction and historical insight visualization. Given the technical scope and design effort required, the workload was effectively split between two members.

### Harsh Gupta

- Led the integration of flight, weather, and airline fleet datasets.
- Focused on time-based and flight traffic features, including:
  - Departure/arrival hour extraction
  - Time-of-day binning
  - Flight distance categorization
  - Route-specific delay metrics
- Built and evaluated Random Forest and FCNN models.
- Developed and implemented the **Prediction tab (Tab 1)** of the dashboard, including user input handling, preprocessing, model inference, and database storage.

### Muthukumaran Ulaganathan

- Focused on cleaning and linking storm events and airfare datasets.
- Engineered storm impact features, including:
  - Total injuries and deaths
  - Severe weather binary flag
  - Damage cost aggregation
- Built and tuned Logistic Regression and XGBoost models.
- Designed and implemented the **History & Insights tab (Tab 2)**, including visualizations (pie charts, line plots, heatmaps), performance metrics, and interactive filters for trend exploration.
