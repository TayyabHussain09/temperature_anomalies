import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import plotly.express as px

# Setting  up Streamlit page configuration
st.set_page_config(page_title="WeatherView", page_icon="🌍", layout="wide")

# Welcome message
st.markdown("""
<div style="text-align: center; padding: 20px; background-color: #f0f8ff; border-radius: 10px;">
    <h1 style="color: #1E90FF; font-family: 'Arial', sans-serif;">Welcome to <b>WeatherView 🌍</b></h1>
    <p style="color: #333; font-size: 18px; font-family: 'Verdana', sans-serif;">
        Dive into the world of climate insights! Analyze global weather patterns, visualize trends, and make data-driven predictions.
    </p>
</div>
""", unsafe_allow_html=True)


# Loading datasets 
global_temp_anomalies = pd.read_csv("global-temperature-anomalies-by-month.csv")
land_temp_state = pd.read_csv("GlobalLandTemperaturesByState.csv")
land_temp_major_city = pd.read_csv("GlobalLandTemperaturesByMajorCity.csv")
land_temp_country = pd.read_csv("GlobalLandTemperaturesByCountry.csv")
global_temperatures = pd.read_csv("GlobalTemperatures.csv")
flood_data = pd.read_csv("flood_data.csv")
marine_data = pd.read_csv("marine_data.csv")
weather_data = pd.read_csv("weather_data.csv")
land_temp_city = pd.read_csv("GlobalLandTemperaturesByCity.csv")
climate_data = pd.read_csv('climate_data.csv')

# Sidebar for navigation
# Sidebar Section Selection
st.sidebar.markdown("""
<div style="text-align: center; padding: 10px; background-color: #e6f7ff; border-radius: 8px; margin-bottom: 15px;">
    <h3 style="color: #007acc; font-family: 'Arial', sans-serif; margin: 0;">Navigation</h3>
    <p style="color: #333; font-size: 14px; margin: 5px 0;">Select a section to explore features</p>
</div>
""", unsafe_allow_html=True)

section = st.sidebar.selectbox(
    "Select Section", 
    [
        "Clean Data", 
        "Summary Statistics", 
        "Pair Plots", 
        "Heatmaps", 
        "Time Series Analysis", 
        "CDA", 
        "Feature Engineering",
        "climate_health_status"
    ]
)


# Clean Data Section
if section == "Clean Data":
    # titile and description
    st.markdown("<h2 style='text-align: center; color: #4CAF50;'>Clean Data (First 50 values of each dataset--scroll down to view full page and select navigation on left to see all section and facts, have fun!)</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Explore the cleaned data, with graphs showcasing the column distributions for better insights.</p>", unsafe_allow_html=True)
    
    # Converting date columns to datetime for all datasets (unchanged)
    date_columns = ['date', 'Date', 'dt']
    for col in date_columns:
        if col in weather_data.columns:
            weather_data[col] = pd.to_datetime(weather_data[col], errors='coerce')
        if col in global_temp_anomalies.columns:
            global_temp_anomalies[col] = pd.to_datetime(global_temp_anomalies[col], errors='coerce')
        if col in land_temp_state.columns:
            land_temp_state[col] = pd.to_datetime(land_temp_state[col], errors='coerce')
        if col in land_temp_major_city.columns:
            land_temp_major_city[col] = pd.to_datetime(land_temp_major_city[col], errors='coerce')
        if col in land_temp_country.columns:
            land_temp_country[col] = pd.to_datetime(land_temp_country[col], errors='coerce')
        if col in global_temperatures.columns:
            global_temperatures[col] = pd.to_datetime(global_temperatures[col], errors='coerce')
        if col in flood_data.columns:
            flood_data[col] = pd.to_datetime(flood_data[col], errors='coerce')
        if col in marine_data.columns:
            marine_data[col] = pd.to_datetime(marine_data[col], errors='coerce')
        if col in land_temp_city.columns:
            land_temp_city[col] = pd.to_datetime(land_temp_city[col], errors='coerce')

    # Displaying first 50 values of cleaned data for all 9 datasets with a small graph showing structure
    datasets = {
        "Weather Data": weather_data,
        "Global Temperature Anomalies": global_temp_anomalies,
        "Land Temperatures by State": land_temp_state,
        "Land Temperatures by Major City": land_temp_major_city,
        "Land Temperatures by Country": land_temp_country,
        "Global Temperatures": global_temperatures,
        "Flood Data": flood_data,
        "Marine Data": marine_data,
        "Land Temperatures by City": land_temp_city
    }
    
    for name, df in datasets.items():
        # Section title with icon and color
        st.markdown(f"<h4 style='color: #2196F3; text-align: center;'>{name} <span style='color: #FFC107;'>📊</span></h4>", unsafe_allow_html=True)
        
        # Display first 50 rows of the dataset
        st.write(df.head(50))
        
        # Showing a small graph for each dataset (distributions of the columns)
        for column in df.select_dtypes(include=['float64', 'int64']).columns:  # Only numeric columns for visualization
            fig, ax = plt.subplots(figsize=(2, 1.5))  # Further reduced figure size
            sns.histplot(df[column], kde=True, ax=ax, color='skyblue')
            ax.set_title(f'Distribution of {column}', fontsize=8)  # Reduced font size for the title
            st.pyplot(fig)

        # Adding a horizontal line for separation
        st.markdown("<hr style='border: 1px solid #4CAF50;'>", unsafe_allow_html=True)

    # spacing
    st.markdown("<br><br><p style='text-align: center; color: gray;'>End of Clean Data Section</p>", unsafe_allow_html=True)


# Summary Statistics Section
elif section == "Summary Statistics":
    # Add styling for the section title and description
    st.markdown("<h2 style='text-align: center; color: #4CAF50;'>Summary Statistics</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Displaying summary statistics for each dataset, including mean, median, mode, and more for better insights.</p>", unsafe_allow_html=True)
    
    # Define the datasets
    datasets = {
        "Weather Data": weather_data,
        "Marine Data": marine_data,
        "Global Temperatures": global_temperatures,
        "Temperature Anomalies": global_temp_anomalies,
        "Land Temperatures by Major City": land_temp_major_city,
        "Land Temperatures by Country": land_temp_country,
        "Land Temperatures by State": land_temp_state,
        "Land Temperatures by City": land_temp_city,
        "Flood Data": flood_data
    }
    
    # Loop through each dataset and display summary statistics
    for name, data in datasets.items():
        st.markdown(f"<h4 style='color: #2196F3; text-align: center;'>{name} <span style='color: #FFC107;'>📊</span></h4>", unsafe_allow_html=True)
        
        # Displaying the descriptive statistics using describe()
        st.write(f"**Descriptive Statistics for {name}:**")
        st.write(data.describe())
        
        # Display mean, median, and mode for each numeric column
        st.write(f"**Mean, Median, and Mode for {name}:**")
        
        # Filter only numeric columns for mean, median, and mode
        numeric_data = data.select_dtypes(include=['float64', 'int64'])

        # Calculate and display mean, median, and mode
        mean_values = numeric_data.mean()
        median_values = numeric_data.median()
        
        # Check if mode exists before accessing
        mode_values = numeric_data.mode()
        mode_values = mode_values.iloc[0] if not mode_values.empty else 'No mode'  # If no mode, set to 'No mode'
        
        # Create a DataFrame to display the mean, median, and mode
        summary_stats = pd.DataFrame({
            'Mean': mean_values,
            'Median': median_values,
            'Mode': mode_values
        })
        
        st.write(summary_stats)
        
        # Adding a horizontal line for separation
        st.markdown("<hr style='border: 1px solid #4CAF50;'>", unsafe_allow_html=True)

    # Optionally, add some spacing and footer for aesthetics
    st.markdown("<br><br><p style='text-align: center; color: gray;'>End of Summary Statistics Section</p>", unsafe_allow_html=True)


# Pair Plots Section
elif section == "Pair Plots":
    st.subheader("Pair Plots")
    st.write("Creating pair plots to visualize relationships, correlations, and distributions between multiple variables in a dataset for data exploration and analysis.")
    
    datasets = {
        "Global Temperature Anomalies": global_temp_anomalies,
        "Land Temperatures by State": land_temp_state,
        "Land Temperatures by Major City": land_temp_major_city,
        "Land Temperatures by Country": land_temp_country,
        "Global Temperatures": global_temperatures,
        "Flood Data": flood_data,
        "Marine Data": marine_data,
        "Weather Data": weather_data,
        "Land Temperatures by City": land_temp_city
    }
    
    for name, data in datasets.items():
        numerical_data = data.select_dtypes(include=['float64', 'int64'])
        if numerical_data.empty:
            st.write(f"No numerical columns found in {name} for pairplot.")
        else:
            st.write(f"Pair plot for {name}")
            sns.pairplot(numerical_data)
            st.pyplot()

# Heatmaps Section
elif section == "Heatmaps":
    st.subheader("Heatmaps")
    st.write("Creating heatmaps is helping to visualize the intensity of relationships, correlations, or patterns in data through color-coded matrices for better analysis , interpretation and future predictions.")
    
    def plot_correlation_matrix(data, title="Correlation Matrix"):
        numerical_data = data.select_dtypes(include=['float64', 'int64'])
        correlation_matrix = numerical_data.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title(title, fontsize=16)
        st.pyplot()

    datasets = {
        "Global Temperature Anomalies": global_temp_anomalies,
        "Land Temperatures by State": land_temp_state,
        "Land Temperatures by Major City": land_temp_major_city,
        "Land Temperatures by Country": land_temp_country,
        "Global Temperatures": global_temperatures,
        "Flood Data": flood_data,
        "Marine Data": marine_data,
        "Weather Data": weather_data,
        "Land Temperatures by City": land_temp_city
    }

    for name, data in datasets.items():
        plot_correlation_matrix(data, title=f'Correlation Matrix - {name}')

# Time Series Analysis Section
elif section == "Time Series Analysis":
    # Add some styling for the section title and description
    st.markdown("<h2 style='text-align: center; color: #4CAF50;'>Time Series Analysis <span style='color: #FFC107;'>📈</span></h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Performing time series analysis on weather data is helping to identify trends, seasonal patterns, and anomalies over time for forecasting and insights, results will be shown in future engineering and climate health section. stay tuned .........</p>", unsafe_allow_html=True)
    
    # Function to plot time series data
    def plot_time_series(data, time_column, value_column, title="Time Series Analysis"):
        data[time_column] = pd.to_datetime(data[time_column], errors='coerce')
        plt.figure(figsize=(10, 6))  # Reduced figure size for better fit
        plt.plot(data[time_column], data[value_column], color='skyblue', linewidth=2)
        plt.title(title, fontsize=14)
        plt.xlabel('Date')
        plt.ylabel(value_column)
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.7)
        st.pyplot()

    # Datasets and their associated icons
    datasets = {
        "Global Temperature Anomalies": (global_temp_anomalies, "🌍"),
        "Land Temperatures by State": (land_temp_state, "🏞️"),
        "Land Temperatures by Major City": (land_temp_major_city, "🏙️"),
        "Land Temperatures by Country": (land_temp_country, "🌏"),
        "Global Temperatures": (global_temperatures, "🌡️"),
        "Flood Data": (flood_data, "🌊"),
        "Marine Data": (marine_data, "🌊"),
        "Weather Data": (weather_data, "🌦️"),
        "Land Temperatures by City": (land_temp_city, "🏙️")
    }

    # Loop through each dataset and plot time series
    for name, (data, icon) in datasets.items():
        if 'Date' in data.columns:
            numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
            for column in numerical_columns:
                st.markdown(f"<h4 style='color: #2196F3; text-align: center;'>{icon} {name} - {column}</h4>", unsafe_allow_html=True)
                plot_time_series(data, 'Date', column, title=f'Time Series - {name} ({column})')
                
                # Adding a horizontal line for separation
                st.markdown("<hr style='border: 1px solid #4CAF50;'>", unsafe_allow_html=True)

    # Optionally, add some spacing and footer for aesthetics
    st.markdown("<br><br><p style='text-align: center; color: gray;'>End of Time Series Analysis Section</p>", unsafe_allow_html=True)


# CDA Section
elif section == "CDA":
    st.subheader("Canonical Discriminant Analysis (CDA)")
    st.write("Performing CDA for each dataset in backend.")
    
    def perform_cda(data, target_column, title="Canonical Discriminant Analysis"):
        le = LabelEncoder()
        data[target_column] = le.fit_transform(data[target_column])
        numerical_data = data.select_dtypes(include=['float64', 'int64'])
        X = numerical_data
        y = data[target_column]
        lda = LinearDiscriminantAnalysis()
        lda_result = lda.fit_transform(X, y)
        plt.figure(figsize=(8, 6))
        plt.scatter(lda_result[:, 0], lda_result[:, 1], c=y, cmap='coolwarm')
        plt.title(f'{title} - 2D Projection', fontsize=16)
        plt.xlabel('LD1')
        plt.ylabel('LD2')
        st.pyplot()

    datasets = {
        "Global Temperature Anomalies": global_temp_anomalies,
        "Land Temperatures by State": land_temp_state,
        "Land Temperatures by Major City": land_temp_major_city,
        "Land Temperatures by Country": land_temp_country,
        "Global Temperatures": global_temperatures,
        "Flood Data": flood_data,
        "Marine Data": marine_data,
        "Weather Data": weather_data,
        "Land Temperatures by City": land_temp_city
    }

    for name, data in datasets.items():
        if 'Category' in data.columns:
            perform_cda(data, 'Category', title=f'CDA - {name}')


# Feature Engineering Section

# Feature Engineering Section
elif section == "Feature Engineering":
    st.subheader("Feature Engineering")
    st.write("Performing feature engineering for climate data.")

    # Load the dataset (replace with your actual file path)
    climate_data = pd.read_csv('climate_data.csv')

    # Check the first few rows of the data
    st.write("First few rows of the dataset:")
    st.write(climate_data.head())

    # 1. Preprocessing: Convert 'date' to datetime format
    climate_data['date'] = pd.to_datetime(climate_data['date'])

    # 2. Extract year and month from the 'date' column for feature engineering
    climate_data['year'] = climate_data['date'].dt.year
    climate_data['month'] = climate_data['date'].dt.month
    climate_data['day'] = climate_data['date'].dt.dayofyear  # Day of the year (1 to 365)

    # 3. Feature Engineering: Create additional features
    # For example, let's create a "temperature range" feature (difference between max and min temperature)
    climate_data['temperature_range'] = climate_data['temperature_2m_max'] - climate_data['temperature_2m_min']

    # You can also add lag features (e.g., temperature of the previous month or year)
    climate_data['temperature_2m_mean_lag1'] = climate_data['temperature_2m_mean'].shift(1)  # Previous month

    # For precipitation, we can calculate the rolling mean (e.g., average precipitation over the last 3 months)
    climate_data['precipitation_rolling_mean'] = climate_data['precipitation_sum'].rolling(window=3).mean()

    # 4. Feature Selection: Use temperature, precipitation, and other relevant columns for prediction
    X = climate_data[['year', 'month', 'temperature_2m_mean', 'temperature_2m_max', 'temperature_2m_min', 'precipitation_sum', 'temperature_range', 'temperature_2m_mean_lag1', 'precipitation_rolling_mean']]
    y = climate_data['temperature_2m_mean']  # Target variable (Mean Temperature)

    # Drop rows with missing values (due to lag and rolling mean)
    X = X.dropna()
    y = y[X.index]  # Ensure y matches the filtered X

    # 5. Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 6. Train the model (Random Forest Regressor)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 7. Make predictions
    y_pred = model.predict(X_test)

    # 8. Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    st.write(f"Mean Absolute Error for Temperature Prediction: {mae}")

    # 9. Plot the predicted vs actual values
    st.write("Actual vs Predicted Temperature:")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.xlabel('Actual Temperature')
    plt.ylabel('Predicted Temperature')
    plt.title('Actual vs Predicted Temperature')
    st.pyplot()

    # 10. Plot Feature Importance
    feature_importance = model.feature_importances_
    st.write("Feature Importance:")
    sns.barplot(x=feature_importance, y=X.columns)
    plt.title('Feature Importance')
    st.pyplot()

    # 11. Predict future temperature (for the next 5 years)
    future_years = pd.DataFrame({'year': [2025, 2026, 2027, 2028, 2029],
                                 'month': [1, 1, 1, 1, 1],
                                 'temperature_2m_mean': [None] * 5,
                                 'temperature_2m_max': [None] * 5,
                                 'temperature_2m_min': [None] * 5,
                                 'precipitation_sum': [None] * 5,
                                 'temperature_range': [None] * 5,
                                 'temperature_2m_mean_lag1': [None] * 5,
                                 'precipitation_rolling_mean': [None] * 5})
    future_temperature_predictions = model.predict(future_years[['year', 'month', 'temperature_2m_mean', 'temperature_2m_max', 'temperature_2m_min', 'precipitation_sum', 'temperature_range', 'temperature_2m_mean_lag1', 'precipitation_rolling_mean']])

    # 12. Visualize future temperature predictions
    st.write("Future Temperature Predictions (2025-2029):")
    plt.figure(figsize=(10, 6))
    plt.plot([2025, 2026, 2027, 2028, 2029], future_temperature_predictions, marker='o')
    plt.title('Future Temperature Predictions (2025-2029)')
    plt.xlabel('Year')
    plt.ylabel('Predicted Temperature')
    st.pyplot()

    # 13. Climate Health Estimation
    climate_health = "Healthy"
    if future_temperature_predictions[-1] - future_temperature_predictions[0] > 2:
        climate_health = "Unhealthy"
    elif future_temperature_predictions[-1] - future_temperature_predictions[0] > 4:
        climate_health = "Critical"

    st.write(f"Climate Health Status: {climate_health}")

    # 14. Suggestions based on Climate Health
    if climate_health == "Healthy":
        st.write("Suggestions: Continue monitoring and focus on sustainability efforts.")
    elif climate_health == "Unhealthy":
        st.write("Suggestions: Take action to reduce emissions, invest in renewable energy, and promote climate adaptation strategies.")
    else:
        st.write("Suggestions: Immediate action required to address climate emergency. Focus on carbon reduction, renewable energy, and large-scale adaptation measures.")


# ----------------------------------------------------------

# section climate_health_status
elif section == "climate_health_status":
    st.title("Climate Health Analysis")

    # Load the datasets
    anomalies_df = pd.read_csv("global-temperature-anomalies-by-month.csv")
    climate_df = pd.read_csv("climate_data.csv")

    # Data preprocessing for anomalies_df
    anomalies_df['Year'] = pd.to_datetime(anomalies_df['Year'], format='%Y')
    anomalies_df['Year'] = anomalies_df['Year'].dt.year
    anomalies_summary = anomalies_df.groupby('Year')['temperature_anomaly'].mean().reset_index()

    # Data preprocessing for climate_df
    climate_df['date'] = pd.to_datetime(climate_df['date'])
    climate_df['Year'] = climate_df['date'].dt.year
    climate_summary = climate_df.groupby('Year').agg({
        'temperature_2m_mean': 'mean',
        'temperature_2m_max': 'mean',
        'temperature_2m_min': 'mean',
        'precipitation_sum': 'sum'
    }).reset_index()

    # Merge datasets for combined analysis
    merged_df = pd.merge(anomalies_summary, climate_summary, on='Year', how='inner')

    # Visualization: Temperature Anomalies Over Time
    st.subheader("Temperature Anomalies Over Time")
    fig1 = px.line(anomalies_summary, x='Year', y='temperature_anomaly',
                   title="Global Temperature Anomalies (Yearly Average)",
                   labels={'temperature_anomaly': 'Temperature Anomaly (°C)', 'Year': 'Year'})
    st.plotly_chart(fig1)

    # Visualization: Climate Trends Over Time
    st.subheader("Climate Trends Over Time")
    fig2 = px.line(climate_summary, x='Year',
                   y=['temperature_2m_mean', 'temperature_2m_max', 'temperature_2m_min'],
                   title="Temperature Trends (Mean, Max, Min)",
                   labels={'value': 'Temperature (°C)', 'Year': 'Year', 'variable': 'Temperature Type'})
    st.plotly_chart(fig2)

    fig3 = px.bar(climate_summary, x='Year', y='precipitation_sum',
                  title="Yearly Precipitation",
                  labels={'precipitation_sum': 'Precipitation (mm)', 'Year': 'Year'})
    st.plotly_chart(fig3)

    # Predictive Analysis
    st.subheader("Predictive Analysis: Temperature Anomalies")
    X = anomalies_summary[['Year']]
    y = anomalies_summary['temperature_anomaly']

    # Train-test split
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict future anomalies
    future_years = pd.DataFrame({'Year': range(2025, 2031)})
    future_predictions = model.predict(future_years)

    # Combine future predictions with existing data
    future_df = pd.DataFrame({'Year': future_years['Year'], 'Predicted_Anomaly': future_predictions})
    st.write("Predicted Temperature Anomalies for 2025-2030:")
    st.dataframe(future_df)

    # Visualization: Future Predictions
    st.subheader("Future Predictions: Temperature Anomalies")
    fig4 = px.line(future_df, x='Year', y='Predicted_Anomaly',
                   title="Predicted Global Temperature Anomalies (2025-2030)",
                   labels={'Predicted_Anomaly': 'Temperature Anomaly (°C)', 'Year': 'Year'})
    st.plotly_chart(fig4)

    # Climate Health Insights
    st.subheader("Insights and Suggestions")
    st.write("""
    - Global temperature anomalies have been steadily increasing, indicating a warming trend.
    - Yearly precipitation patterns show variability, which could impact agriculture and water resources.
    - Predictive analysis suggests further warming in the next decade. 
    - Governments and organizations should focus on reducing greenhouse gas emissions and adopting sustainable practices.
    """)




    


