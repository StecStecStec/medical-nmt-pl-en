import csv

with open('data_testing_short.csv', 'r', encoding='utf-8', newline='') as file:
    reader = csv.reader(file)
    line_count = sum(1 for row in reader)

print(f"The file has {line_count} lines.")
