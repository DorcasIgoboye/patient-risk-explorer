import os, csv, random, math

# Path to save dataset
DATA = os.path.join("data", "synthetic_patients.csv")
os.makedirs("data", exist_ok=True)

def generate_data(path=DATA, N=500):
    random.seed(42)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        hdr = ["age","bmi","glucose","family_history","activity_hours",
               "systolic","smoking","ldl","diabetes","heart"]
        writer.writerow(hdr)

        for _ in range(N):
            age = int(max(18, min(90, random.gauss(50, 15))))
            bmi = round(max(15, min(50, random.gauss(28, 6))), 1)
            glucose = int(max(60, min(200, random.gauss(100, 20))))
            family = 1 if random.random() < 0.25 else 0
            activity = round(max(0, min(20, random.gauss(3, 2))), 1)
            systolic = int(max(90, min(200, random.gauss(130, 15))))
            smoking = 1 if random.random() < 0.18 else 0
            ldl = int(max(50, min(240, random.gauss(120, 30))))

            # risk signals
            ds = 0.02*age + 0.08*bmi + 0.06*(glucose-90) + 0.9*family - 0.05*activity + random.gauss(0,0.6)
            hs = 0.025*age + 0.06*bmi + 0.03*(systolic-120) + 0.9*smoking + 0.02*(ldl-100) + random.gauss(0,0.7)

            dp = 1.0/(1.0+math.exp(-ds/10.0))
            hp = 1.0/(1.0+math.exp(-hs/10.0))

            diabetes = 1 if dp > 0.55 else 0
            heart = 1 if hp > 0.55 else 0

            writer.writerow([age,bmi,glucose,family,activity,systolic,smoking,ldl,diabetes,heart])

    print(f"Wrote {N} rows to {path}")

if __name__ == "__main__":
    generate_data()
