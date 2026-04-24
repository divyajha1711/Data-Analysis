import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#LOAD DATASET
data = pd.read_csv(r"C:\Users\DELL\Downloads\BKB_WaterQualityData_2020084.csv")

# VIEW DATA STRUCTURE AND SUMMARY
print(data.info())#CHECK DATATYES
print(data.describe())
print(data.head()) #helps to load the first 5 rows of the dataset


# CHECK THE TOTAL NUMBER OF ROWS AND COLUMNS 
print("Number of rows and columns:",data.shape)

#DATA CLEANING

#NOW CHECK THE COLUMNS
print(data.columns)

#DELETING UNWANTED COLUMNS
data = data.drop(columns=["WhoVerified", "DateVerified", "Unit_Id","Air Temp (?F)","AirTemp (C)"])

#REMOVING DUPLICATE ROWS
duplicates = data.duplicated().sum()
data = data.drop_duplicates()
print("Duplicate rows:", duplicates)

#RENAMING COLUMNS
data.rename(columns={'pH (standard units)' : 'pH'}, inplace=True)
data.rename(columns={'Salinity (ppt)' : 'Salinity'}, inplace=True)
data.rename(columns={'Dissolved Oxygen (mg/L)' : 'DissolvedOxygen'}, inplace=True)
data.rename(columns={'Water Temp (?C)' : 'Water_Temp'}, inplace=True)
data.rename(columns={'Water Depth (m)' : 'Water_Depth'}, inplace=True)
data.rename(columns={'Secchi Depth (m)' : 'Secchi_Depth'}, inplace=True)



#CHECK MISSING DATAS
print(data.isnull().sum())

#REMOVING ROWS WITH NULL VALUES
data = data.dropna()

#NOW CHECKING THE MISSISNG VALUES AGAIN
print(data.isnull().sum())

#OBJECTIVES

#OBJECTIVE 1: To analyze the distribution of data using histogram plots.
#Histogram of pH
plt.figure(figsize=(8,5))
plt.hist(data["pH"], bins=25, color='skyblue', edgecolor='black')
plt.title("Histogram of pH Values", color='darkblue')
plt.xlabel("pH", color='green')
plt.ylabel("Frequency", color='green')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

#Histogram of salinity
plt.figure(figsize=(8,5))
plt.hist(data["Salinity"], bins=15,color='red')
plt.title("Histogram of Salinity Values" ,color='purple')
plt.xlabel("Salinity", color='black')
plt.ylabel("Frequency", color='black')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

#Histogram of DissolvedOxygen
plt.figure(figsize=(8,5))
plt.hist(data["DissolvedOxygen"], bins=20, color='orange' , edgecolor='black')
plt.title("Distribution of Dissolved Oxygen", color='darkblue')
plt.xlabel("Dissolved Oxygen", color='green')
plt.ylabel("Frequency", color='green')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


#OBJECTIVE 2: To Analyze Relationship Between pH and Dissolved Oxygen
plt.figure(figsize=(8,5))
plt.scatter(data["pH"], data["DissolvedOxygen"], color='red')
plt.title("pH vs Dissolved Oxygen", color='green')
plt.xlabel("pH", color='blue')
plt.ylabel("Dissolved Oxygen", color='blue')
plt.show()
# the scatter plots shows that the points are loosely packed and are in one direction 
# the correlation among them is weak.


#OBJECTIVE 3:Implementation of Linear Regression for Predictive Analysis among water temp and dissolved oxygen
# Prepare data
X = data[["Water_Temp"]].dropna()
y = data.loc[X.index, "DissolvedOxygen"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict
y_pred = model.predict(X)

# Calculate MSE
mse = mean_squared_error(y, y_pred)
print("Mean Squared Error:", mse)

# Plot
plt.scatter(X, y, color='purple')   # scatter color
plt.plot(X, y_pred, color='red', linewidth=2)   # regression line with width
plt.title("Linear Regression: Temp vs DO", color='darkblue')
plt.xlabel("Temperature", color='green')
plt.ylabel("Dissolved Oxygen", color='green')
plt.show()


#OBJECTIVE 4: Year-wise Trend Analysis
year_analysis = data.groupby("Year")["DissolvedOxygen"].mean()
plt.plot(year_analysis.index, year_analysis.values,
         color='purple', linewidth=2, marker='o')  
plt.title("Year-wise Dissolved Oxygen Trend", color='darkblue')
plt.xlabel("Year", color='green')
plt.ylabel("Average DO", color='green')
plt.grid(True, linestyle='--', alpha=0.6)  
plt.show()
#the graph is this way because their is no entry of the year 2015
data["Year"].value_counts() #tells me how many entries per year

#OBJECTIVE 5: Outlier Detection using Boxplot
# pH Boxplot
plt.boxplot(data["pH"].dropna(),
            patch_artist=True,
            boxprops=dict(facecolor='lightgreen', linewidth=4),
            whiskerprops=dict(color='purple', linewidth=4),
            capprops=dict(color='green', linewidth=4),
            medianprops=dict(color='red', linewidth=4))
plt.title("Boxplot of pH", color='darkblue')
plt.ylabel("pH", color='green')
plt.show()

# Dissolved Oxygen Boxplot
plt.boxplot(data["DissolvedOxygen"].dropna(),
            patch_artist=True,
            boxprops=dict(facecolor='lightblue', linewidth=4),
            whiskerprops=dict(color='green', linewidth=4),
            capprops=dict(color='blue', linewidth=4),
            medianprops=dict(color='red', linewidth=4))
plt.title("Boxplot of Dissolved Oxygen", color='darkblue')
plt.ylabel("Dissolved Oxygen", color='green')
plt.show()


#OBJECTIVE 6: Correlation Analysis Between Parameters
# Correlation matrix
corr = data.corr(numeric_only=True)
plt.figure(figsize=(10,6))

# Heatmap
plt.imshow(corr, cmap='coolwarm')
plt.colorbar()

# Add values inside boxes
for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        plt.text(j, i, round(corr.iloc[i, j], 2),
                 ha='center', va='center', color='black')

# Labels
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Correlation Heatmap", color='darkblue')
plt.show()


#OBJECTIVE 7: Water Quality Classification
def water_quality(ph):
    if ph < 6.5:
        return "Acidic"
    elif ph <= 8.5:
        return "Normal"
    else:
        return "Alkaline"

data["Water_Quality"] = data["pH"].apply(water_quality)

# Count categories
data["Water_Quality"].value_counts().plot(kind="bar")
plt.title("Water Quality Categories")
plt.xlabel("Category")
plt.ylabel("Count")
plt.show()

