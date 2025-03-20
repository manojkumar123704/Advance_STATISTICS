import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import scipy.stats as stats

st.title("Sales Data Analysis for Retail Store using statistics")
st.write("This application analyzes sales data for various product categories.")

def generate_data():
    np.random.seed(42)
    data = {
        'product_id': range(1, 21),
        'product_name': [f'Product {i}' for i in range(1, 21)],
        'category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Sports'], 20),
        'units_sold': np.random.poisson(lam=20, size=20),
        'sale_date': pd.date_range(start='2023-01-01', periods=20, freq='D')
    }
    return pd.DataFrame(data)
sales_data = generate_data()
# display the sales data
st.subheader("Sales Data")
st.write(sales_data)  

st.header("Descriptive Statistics")
descriptive_stats = sales_data['units_sold'].describe()
st.write(descriptive_stats)

st.header("The mean of the dataset")
meansales=sales_data['units_sold'].mean()
st.write(meansales)

st.header("The median of the dataset")
meadiansales=sales_data['units_sold'].median()
st.write(meadiansales)

st.header("The standard deviation of the dataset")
standarddeviation=sales_data['units_sold'].std()
st.write(standarddeviation) 

st.title("Category Statistics")
category_stats = sales_data.groupby('category')['units_sold'].agg(['sum', 'mean', 'std']).reset_index()     
category_stats.columns = ['Category', 'Total Units Sold', 'Average Units Sold', 'Std Dev of Units Sold']
st.write(category_stats)
st.title("Inferential Statistics")
confidence_level = 0.95
degrees_freedom = len(sales_data['units_sold']) - 1
sample_mean = meansales
sample_standard_error = sales_data['units_sold'].std() /np.sqrt(len(sales_data['units_sold']))
t_score = stats.t.ppf((1+confidence_level)/2,degrees_freedom)
margin_error=t_score*sample_standard_error
confidence_interval=(sample_mean-margin_error,sample_mean+margin_error)
st.write(f"confidence level:{confidence_level}")
st.write(f"degrees of freedom:{degrees_freedom}")
st.write(f"sample mean:{sample_mean}")
st.write(f"sample standard error:{sample_standard_error}")
st.write(f"t-score:{t_score}")
st.write(f"margin of error:{margin_error}") 
st.write(f"confidence interval:{confidence_interval}")
st.title("Hypothesis Testing")
st.write("Let's test the hypothesis that the average units sold is greater than 20")    
null_hypothesis = "The average units sold is 20"
alternative_hypothesis = "The average units sold is greater than 20"
t_score,p_value = stats.ttest_1samp(sales_data['units_sold'],20)
if p_value<0.05:
    st.write("we reject the null hypothesis")
else:
    st.write("we fail to reject the null hypothesis")
st.write(f"t-score:{t_score}")
st.write(f"p-value:{p_value}")
st.title("Data Visualization")
st.write("Let's visualize the total units sold by category")    
plt.figure(figsize=(10, 6))
sns.barplot(x='Category', y='Total Units Sold', data=category_stats)
plt.title("Total Units Sold by Category")
plt.xlabel("Category")
plt.ylabel("Total Units Sold")
st.pyplot(plt)  
plt.boxplot(sales_data['units_sold'])
plt.title("Units Sold Distribution")
plt.xlabel("Units Sold")
plt.ylabel("Frequency")
st.pyplot(plt)
st.title("Time Series Analysis")
st.write("Let's analyze the trend of units sold over time") 
sales_data['sale_date'] = pd.to_datetime(sales_data['sale_date'])
time_series_data = sales_data.groupby('sale_date')['units_sold'].sum()
plt.figure(figsize=(10, 6))
plt.boxplot(time_series_data)
plt.title("Units Sold Over Time")
plt.xlabel("Date")
plt.ylabel("Units Sold")
st.pyplot(plt)
st.title("Conclusion")  
st.write("In this analysis, we explored the sales data for a retail store. We calculated descriptive statistics, category statistics, and performed inferential statistics to test a hypothesis. We also visualized the data to gain insights into the sales performance. The analysis provides valuable information for decision-making and future planning.")
st.write("Thank you for using this application!")