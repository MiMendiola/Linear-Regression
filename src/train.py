import csv
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

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

        # Normalize and get the standard desviation on the mileages
        total_mileages = 0
        mileages_mean = sum(mileages) / len(mileages)
        for mileage in mileages:
            total_mileages += (mileage - mileages_mean) ** 2
        
        mileage_std = (total_mileages / len(mileages)) ** 0.5
        
        for mileage in mileages:
            normalized_mileages.append((mileage - mileages_mean) / mileage_std)

        # Normalize and get the standard desviation on the prices
        total_prices = 0
        prices_mean = sum(prices) / len(prices)
        for price in prices:
            total_prices += (price - prices_mean) ** 2
        
        price_std = (total_prices / len(prices)) ** 0.5
        
        for price in prices:
            normalized_prices.append((price - prices_mean) / price_std)

        # Gradient Descent
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

        # Desnormalized the data
        theta1 = price_std * theta1 / mileage_std
        theta0 = prices_mean + price_std * theta0 - theta1 * mileages_mean
            
        # Save the training
        model_data = {
            "theta0": theta0,
            "theta1": theta1
        }
        with open("model.json", 'w') as file:
            json.dump(model_data, file, indent=4)

        # Bonus
        total_error_mse = 0.0
        total_error_mae = 0.0
        error_variation = 0.0

        for i in range(m):
            prediction_error = theta0 + (theta1 * mileages[i])
            # MSE (MEAN SQUARED ERROR)
            total_error_mse += float(((prediction_error) - prices[i]) ** 2)

            # MAE (Mean Absolute Error)
            total_error_mae += abs(float((prediction_error) - prices[i]))

            # (R-squared) = 1 - (error variation / total variation)
            error_variation += (prices[i] - float((prediction_error))) ** 2

        # MSE (MEAN SQUARED ERROR)
        mse = total_error_mse / m

        # RMSE (Root MSE) == √MSE
        rmse = mse ** 0.5

        # MAE (Mean Absolute Error)
        mae = total_error_mae / m

        # R2 (R-squared)
        r2 = 1 - (error_variation / total_prices)

        # Mostrar resultados en consola
        print("\n--- Métricas de Rendimiento del Modelo ---")
        print(f"MSE  (Mean Squared Error):  {mse:.4f}")
        print(f"RMSE (Root MSE):            {rmse:.4f}")
        print(f"MAE  (Mean Absolute Error): {mae:.4f}")
        print(f"R²   (R-squared):           {r2:.4f}\n")
        

        # Graphics
        # 1. Get all dots of the graphic
        plt.scatter(mileages, prices, color="blue")

        # 2. Get the max and min of mileages and get the end points of the prediction line
        max_mileage = max(mileages)
        min_mileage = min(mileages)

        pred_max_mileage = theta0 + (theta1 * max_mileage)
        pred_min_mileage = theta0 + (theta1 * min_mileage)

        # 3. Draw the regresion line
        plt.plot([min_mileage, max_mileage],[pred_min_mileage, pred_max_mileage], color="red")

        # 4. Add labels and titles
        plt.title("Lineal Regresion: Prices vs Mileages")
        plt.xlabel("Mileage (KM)")
        plt.ylabel("Prices (€)")

        # 5. Create the graph
        plt.savefig("./linear_regression.png", dpi=300)
        print(f"Graph exported correctly")

        # 6. Show the graph
        plt.show()


        # Other bonus
        numpy_theta1, numpy_theta0 = np.polyfit(mileages, prices, 1)
        print("\nNumpy.polyfit compare")
        print(f"SUBJE: {theta0:.4f}, {theta1:.4f}")
        print(f"NUMPY: {numpy_theta0:.4f}, {numpy_theta1:.4f}")

    else:
        print("File not exist") 

if __name__ == "__main__":
    main()
