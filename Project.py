import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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




