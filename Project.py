import numpy as np
import pandas as pd

sales = pd.read_csv("sales.csv")
customers = pd.read_csv("customers.csv")
orders = pd.read_csv("orders.csv")
products = pd.read_csv("products.csv")

print(sales.head())
print(customers.head())
print(orders.head())
print(products.head())

cust_orders = pd.merge(left=customers, right=orders,
                      left_index=True, right_index=True)

cop_data = pd.merge(left=cust_orders, right=products,
                    left_index=True, right_index=True)

print(cop_data)

# Get the number of missing data points per column
missing_values_count = cop_data.isnull().sum()
missing_values_count[:10]

print(missing_values_count)



cop_data["order_date"], cop_data["delivery_date"] = pd.to_datetime(cop_data["order_date"]), pd.to_datetime(cop_data["delivery_date"])
cop_data.info()

print(sales.describe())


