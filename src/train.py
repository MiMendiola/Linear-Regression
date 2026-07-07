import csv
import json
from pathlib import Path

def main():
    data_path = Path('./data.csv')
    mileages = []
    normalized_mileages = []
    prices = []
    normalized_prices = []
    theta0 = 0.0
    theta1 = 0.0

    if data_path.exists():
        with data_path.open('r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                mileages.append(float(row["km"]))
                prices.append(float(row["price"]))

        # 
        total_mileages = 0
        mileages_mean = sum(mileages) / len(mileages)
        for mileage in mileages:
            total_mileages += (mileage - mileages_mean) ** 2
        
        mileage_std = (total_mileages / len(mileages)) ** 0.5
        
        for mileage in mileages:
            normalized_mileages.append((mileage - mileages_mean) / mileage_std)


        total_prices = 0
        prices_mean = sum(prices) / len(prices)
        for price in prices:
            total_prices += (price - prices_mean) ** 2
        
        price_std = (total_prices / len(prices)) ** 0.5
        
        for price in prices:
            normalized_prices.append((price - prices_mean) / price_std)


        m = len(normalized_mileages)
        iterations = 1000
        learningRate = 0.1
        for _ in range(iterations):
            tmp_theta0 = 0
            tmp_theta1 = 0
            for i in range(m):
                prediction = theta0 + (theta1 * normalized_mileages[i])

                tmp_theta0 += prediction - normalized_prices[i]
                tmp_theta1 += (prediction - normalized_prices[i]) * normalized_mileages[i]


            theta0 -= learningRate * (tmp_theta0 / m)
            theta1 -= learningRate * (tmp_theta1 / m)

        
        theta1 = price_std * theta1 / mileage_std
        theta0 = prices_mean + price_std * theta0 - theta1 * mileages_mean
            
        model_data = {
            "theta0": theta0,
            "theta1": theta1
        }

        with open("model.json", 'w') as file:
            json.dump(model_data, file, indent=4)

    else:
        print("File not exist")

if __name__ == "__main__":
    main()
