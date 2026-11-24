
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, col, count
from pyspark.sql.functions import ( sum as sparksum, max, min,avg, countDistinct, mean, stddev, round as spark_round)
# 1. Create SparkSession
spark = SparkSession.builder \
    .appName("Day1Learning") \
    .master("local[*]") \
    .getOrCreate()

print("Spark started successfully!")

# 2. Create employee data
data = [
    (101, "Arun", "IT", 4000),
    (102, "Rahul", "HR", 3000),
    (103, "John", "IT", 5000),
    (104, "David", "Finance", 3500),
    (105, "Sam", "HR", 4500)
]

columns = ["EmpID", "Name", "Department", "Salary"]

# 3. Create DataFrame
df = spark.createDataFrame(data, columns)

# 4. Display DataFrame
df.show()

df.count()
# 5. Display schema
df.printSchema()

# 6. Display selected columns
df.select("Name", "Department").show()
df.show(3)

# 7. Filter employees
df.filter(df.Salary > 3500).show()
# 8. Stop Spark
#8. adding column Bonus
df1 = df.withColumn("Bonus", col("Salary")*0.1)
df1.show()
#9. modifying column salary
df2= df.withColumn("Salary", col("Salary") + 1000)
df2.show()

df.groupBy("Department").agg(avg("Salary").alias("Average_Salary"), count("EmpID").alias("Employee_Count"),min("Salary").alias("Min_Salary"),max("Salary").alias("Max_Salary")).show()

spark.stop()
