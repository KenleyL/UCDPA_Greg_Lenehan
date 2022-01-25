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

#summary od
print(sales.describe())

#Graph

#converting the order date into Month data
cop_data['month_order'] = cop_data['order_date'].dt.month

#sales data for graph
cop_data["sales"] = cop_data["price"] * cop_data["quantity"]

sum_month_order = cop_data.groupby(["month_order"]).sum().astype("int")
plt.figure(figsize=(24, 10))

sns.barplot(
    x=sum_month_order.index, # x-axis
    y=sum_month_order["sales"], # y-axis
    data=sum_month_order, # data
    palette="deep" # palette
)

plt.show()

#dropping duplicates. No duplicates in data
drop_duplicates = cop_data.drop_duplicates()
print(cop_data.shape,drop_duplicates.shape)

mach_learn = cop_data[["sales", "product_ID"]]
print(mach_learn)

X = cop_data[["sales", "order_id"]]
y = cop_data["product_ID"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(len(X_train))

clf = LinearRegression()
print(clf.fit(X_train, y_train))

print(clf.predict(X_test))

print(clf.score(X_test, y_test))



