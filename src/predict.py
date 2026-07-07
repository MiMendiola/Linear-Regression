import json
from pathlib import Path

def main():
    theta0 = 0.0
    theta1 = 0.0
    model_path = Path('./model.json')

    try:
        mileage = float(input("Mileage (KM): "))

        if mileage < 0:
            raise SystemExit("Error: mileage cannot be negative")
        
    except ValueError as error:
        raise SystemExit("Error: mileage must be only numbers") from error
        
    if model_path.exists():
        with model_path.open('r') as file:
            model = json.load(file)
            theta0 = float(model["theta0"])
            theta1 = float(model["theta1"])
        
    estimatePrice = theta0 + (theta1 * mileage)
    print("Estimated price:", f"{estimatePrice:.2f}")

if __name__ == "__main__":
    main()
