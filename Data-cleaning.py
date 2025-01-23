
################ Data cleaning the Iris dataset #################
from sklearn import datasets
import pandas as pd

# load iris dataset
iris = datasets.load_iris()
# Since this is a bunch, create a dataframe
iris_df=pd.DataFrame(iris.data)
iris_df['class']=iris.target

iris_df.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
#### ===> TASK 1: here - add two more lines of the code to find the number and mean of missing data
# Count the number of missing values in each column
missing_counts = iris_df.isnull().sum()

# Calculate the mean of the missing values in the dataset
mean_missing = missing_counts.mean()

print("Missing Counts per Column:\n", missing_counts)
print("Mean Number of Missing Values:", mean_missing)

cleaned_data = iris_df.dropna(how="all", inplace=True) # remove any empty lines


iris_X=iris_df.iloc[:5,[0,1,2,3]]
print(iris_X)

### TASK2: Here - Write a short readme to explain above code and how we can calculate the corrolation amoung featuers with description
# Calculate correlation matrix
import seaborn as sns
import matplotlib.pyplot as plt
correlation_matrix = iris_df.corr()

# Plotting the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix of Iris Dataset Features')
plt.show()