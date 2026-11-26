"""Small, openly shareable *synthetic* fixture: not the hackathon dataset."""
import csv
from pathlib import Path

ROWS = [
    ("The staff were friendly and the room was spotless", 1),
    ("A wonderful stay with helpful reception staff", 1),
    ("Excellent breakfast and a very comfortable bed", 1),
    ("Clean hotel with great service and quiet rooms", 1),
    ("I enjoyed the lovely view and helpful team", 1),
    ("Fantastic visit and good facilities", 1),
    ("The room was pleasant and the location convenient", 1),
    ("Friendly service made the trip enjoyable", 1),
    ("Dirty room and broken shower", 0),
    ("Terrible service with rude reception", 0),
    ("The hotel was noisy and the bed uncomfortable", 0),
    ("Disappointing breakfast and filthy bathroom", 0),
    ("Poor location and unhelpful staff", 0),
    ("Awful stay with a leaking ceiling", 0),
    ("Bad experience and unpleasant smell in the room", 0),
    ("The room was cold and service was slow", 0),
]

def main():
    path = Path(__file__).resolve().parents[1] / "examples" / "synthetic_reviews.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Review", "Promotore"])
        writer.writerows(ROWS)
    print(f"Created {path} (synthetic only; NOT evidence of hackathon accuracy).")

if __name__ == "__main__":
    main()
