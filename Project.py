import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#importing the 4 csv files
sales = pd.read_csv("sales.csv")
customers = pd.read_csv("customers.csv")
orders = pd.read_csv("orders.csv")
products = pd.read_csv("products.csv")

#to see the first 5 rows
print(sales.head())
print(customers.head())
print(orders.head())
print(products.head())

#to know the names of the columns
print(sales.columns)
print(customers.columns)
print(orders.columns)
print(products.columns)

#merging the sales, customers & Orders dataframes
cust_orders = pd.merge(left=customers, right=orders,
                      left_index=True, right_index=True)

cop_data = pd.merge(left=cust_orders, right=products,
                    left_index=True, right_index=True)

print(cop_data)

# Get the number of missing data points per column
missing_values_count = cop_data.isnull().sum()
missing_values_count[:10]

print(missing_values_count)



#convert order date and delivery date columns to datetime function
cop_data["order_date"], cop_data["delivery_date"] = pd.to_datetime(cop_data["order_date"]), pd.to_datetime(cop_data["delivery_date"])
cop_data.info()



#Graph

#converting the order date into Month data
cop_data['month_order'] = cop_data['order_date'].dt.month

#sales data for graph
cop_data["sales"] = cop_data["price"] * cop_data["quantity"]

sum_month_order = cop_data.groupby(["month_order"]).sum().astype("int")
#to see sales data from sum_month_order function
print(sum_month_order)

plt.figure(figsize=(24, 10))

sns.barplot(
    x=sum_month_order.index, # x-axis
    y=sum_month_order["sales"], # y-axis
    data=sum_month_order, # data
    color="blue" # color
)

plt.title("How have Sales changed over the last few Months")
plt.xlabel("Months", color="green", fontsize=20, loc="center")
plt.xticks()
plt.ylabel("Sales in Australian $", color="pink", fontsize=20)
plt.yticks()

plt.show()

top_20_city = (cop_data.groupby("city") # groupping
                      .sum() # sum
                      .astype("int")["sales"] # change type into int and get the sales features
                      .sort_values(ascending=False) # sort values
                      .head(20) # head
                      .to_frame()) # change it into data frame

print(top_20_city)

plt.figure(dpi=100, figsize=(24, 10)) # figuring the size
sns.barplot( # barplot
    x="sales", # x-axis
    y=top_20_city.index, # y-axis
    data=top_20_city, # data
    palette="crest") # palette
# title
plt.title("Top 20 Cities with highest number of sales")
# x-label
plt.xlabel("Sales in Dollar Australia ($)", color="blue")
# y-label
plt.ylabel("Name of Cities", color="blue")

plt.show()

#dropping duplicates. No duplicates in data
drop_duplicates = cop_data.drop_duplicates()
print(cop_data.shape,drop_duplicates.shape)

#the median of total sales
stats = np.array([727160, 611133, 759620, 653023, 552995, 658699, 706053, 688716, 651023, 524515])
print(np.median(stats))

#iterator example
value = stats.__iter__()
item1 = value.__next__()
print(item1)

item2 = value.__next__()
print(item2)



mach_learn = cop_data[["product_ID", "age", "sales"]]
mach_learn.drop_duplicates()
print(mach_learn)

X = mach_learn[["product_ID", "age"]]
y = mach_learn["sales"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
print(len(X_train))


clf = LinearRegression()
print(clf.fit(X_train, y_train))

print(clf.predict(X_test))

print(clf.score(X_test, y_test))

print(re.findall(r"Male",  "Female	Genderfluid	Polygender	Bigender	Polygender	Genderfluid	Bigender	Agender	Male	Bigender	Agender	Bigender	Male	Bigender	Genderfluid	Male	Genderfluid	Male	Agender	Genderqueer	Non-binary	Non-binary	Female	Genderfluid	Genderqueer	Genderqueer	Polygender	Non-binary	Male	Female	Non-binary	Genderqueer	Male	Non-binary	Male	Genderqueer	Male	Male	Genderqueer	Polygender	Male	Genderqueer	Non-binary	Bigender	Male	Agender	Polygender	Male	Female	Genderqueer"))




