Results

The main FedProx run had a test accuracy of 0.728 and a macro F1 of 0.633. The
model performed best on pedestrians and had the most trouble with static
obstacles.

The model relied mostly on the camera input. Removing vision reduced macro F1
from 0.633 to 0.158. Removing radar or motion had almost no effect, and removing
depth caused a smaller drop.

Performance decreased as more inputs were removed. The epistemic uncertainty
score did not increase along with the errors, so it was not a reliable warning
for missing sensor information in these runs.

The current FedAvg and FedProx results are close. More matched runs are needed
before saying that one works better than the other.
